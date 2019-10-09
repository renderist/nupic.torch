# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import argparse
import os
import re
from pathlib import Path
import time
import datetime
import platform
import subprocess
import psutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from nupic.torch.models.sparse_cnn import gsc_sparse_cnn, gsc_super_sparse_cnn
from nupic.torch.modules import rezero_weights, update_boost_strength

from audio_transforms import (
    AddNoise,
    ChangeAmplitude,
    ChangeSpeedAndPitchAudio,
    DeleteSTFT,
    FixAudioLength,
    FixSTFTDimension,
    LoadAudio,
    StretchAudioOnSTFT,
    TimeshiftAudioOnSTFT,
    ToMelSpectrogram,
    ToMelSpectrogramFromSTFT,
    ToSTFT,
    ToTensor,
    Unsqueeze
)


os.chdir(os.path.dirname(os.path.abspath(__file__)))

LEARNING_RATE = 0.01
LEARNING_RATE_GAMMA = 0.9
MOMENTUM = 0.0
EPOCHS = 30
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000
REDUCE_LR_ON_PLATEAU = False

LABELS = tuple(["unknown", "silence", "zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine"])

DATAPATH = Path("data")
EXTRACTPATH = DATAPATH/"raw"


def profile(model, loader, device):
    """
    Evaluate trained model using given dataset loader.
    Called on every epoch.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    :param desc: Description for progress bar
    :type desc: str
    :return: Dict with "accuracy", "loss" and "total_correct"
    """
    global first_time
    model.eval()
    batch_size = loader.batch_size
    num_batches = len(loader)
    times = np.zeros(num_batches)
    global output
    idx = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            tim1 = time.perf_counter()
            output = model(data)
            first_time = False
            tim2 = time.perf_counter()
            tim = (tim2 - tim1)*1.e6 # Convert from s to us
            times[idx] = tim
            idx = idx + 1
            # sum up batch loss
            #loss += criterion(output, target, reduction='sum').item()
            # get the index of the max log-probability
            #pred = output.argmax(dim=1, keepdim=True)
            #total_correct += pred.eq(target.view_as(pred)).sum().item()
    avg_time = times.mean()/batch_size
    med_time = np.median(times)/batch_size # for some reason times.median() provokes a no attribute 'median' error
    std_time = times.std()/batch_size
    min_time = times.min()/batch_size
    max_time = times.max()/batch_size
    #stimes = np.copy(times)
    #stimes.sort()
    #stimes = stimes/batch_size
    #times = times/batch_size
    #return [batch_size, num_batches, times, stimes, avg_time, med_time, std_time, min_time, max_time]
    return [batch_size, num_batches, avg_time, med_time, std_time, min_time, max_time]

def do_profile_test(model, device, batch_size, dataset_name="test", inflation_multiplier=1, num_workers=0):
    """
    Test on the noisy data.

    :param model: pytorch model to be tested
    :type model: torch.nn.Module

    :param device:
    :type device: torch.device
    """
    results = []
    for noise in [0.0]:
        noise_wavdata_to_tensor = [LoadAudio(),
                                   FixAudioLength(),
                                   AddNoise(noise),
                                   ToMelSpectrogram(n_mels=32),
                                   ToTensor("mel_spectrogram", "input"),
                                   Unsqueeze("input")]
        if inflation_multiplier != 1:
            cachefile = "gsc_" + dataset_name + "_noise{}_{}.npz".format("{:.2f}".format(noise)[2:],inflation_multiplier)
        else:
            cachefile = "gsc_" + dataset_name + "_noise{}.npz".format("{:.2f}".format(noise)[2:])
        test_dataset = dataset_from_wavfiles(EXTRACTPATH/dataset_name,
                                             noise_wavdata_to_tensor,
                                             cachefilepath=DATAPATH/cachefile, silence_percentage=0.0, inflation_multiplier=inflation_multiplier)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, drop_last=True, num_workers=num_workers)
        res = profile(model=model, loader=test_loader, device=device)
        # res = [noise] + res
        results = results + res
        #print("Noise level: {}, Results: {}".format(noise, results))
    return results



def dataset_from_wavfiles(folder, wavdata_to_tensor, cachefilepath,
                          silence_percentage=0.0, inflation_multiplier=1):
    """
    Get and cache a processed dataset from a folder of wav files.

    :param folder:
    Folder containing wav files in subfolders, for example "./label1/file1.wav"
    :type folder: pathlib.Path

    :param wavdata_to_tensor:
    List of callable objects that create a tensor from a wav file path when
    called in succession.
    :type wavdata_to_tensor: list

    :param cachefilepath:
    Location to save the processed data.
    :type cachefilepath: pathlib.Path

    :param silence_percentage:
    Controls the number of silence wav files that are appended to the dataset.
    :type silence_percentage: float

    :return: torch.utils.data.TensorDataset
    """
    if cachefilepath.exists():
        x, y = np.load(cachefilepath).values()
        x, y = map(torch.tensor, (x, y))
    else:
        label_to_id = {label: i for i, label in enumerate(LABELS)}

        wavdatas = []
        ids = []

        for label in os.listdir(folder):
            if label.startswith("_"):
                continue

            for f in os.listdir(folder/label):
                d = { "path": folder/label/f }
                for i in range(0, inflation_multiplier):
                    wavdatas.append(d)
                    ids.append(label_to_id[label])

        if silence_percentage > 0.0:
            num_silent = int(len(wavdatas) * silence_percentage)
            for _ in range(num_silent):
                d = {"path": None}
                wavdatas.append(d)
                ids.append(label_to_id["silence"])

        x = torch.zeros(len(wavdatas), 1, 32, 32)
        for i, d in enumerate(tqdm(wavdatas, leave=False,
                                   desc="Processing audio")):
            for xform in wavdata_to_tensor:
                d = xform(d)
            x[i,0] = d
        y = torch.tensor(ids)

        print("Caching data to {}".format(cachefilepath))
        np.savez(cachefilepath, x.numpy(), y.numpy())

    return torch.utils.data.TensorDataset(x, y)

def get_processor_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        return subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip().decode()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line, 1)
    return ""

def print_layer(self, input, output):
    global first_time
    if first_time:
        print('Inside ' + self.__class__.__name__ + ' forward', 'Input Size:', input[0].size(), 'Output Size:', output.data.size())

def print_kwinner2d_1(self, input, output):
    global kwinner2d_1_out_var
    global first_time
    if first_time:
        print_layer(self, input, output)
    if kwinner2d_1_out_var.size()[0] == 0:
        kwinner2d_1_out_var = (output.data != 0).int()
    else:
        kwinner2d_1_out_var.add_((output.data != 0).int())

def print_kwinner2d_2(self, input, output):
    global kwinner2d_2_out_var
    global first_time
    if first_time:
        print_layer(self, input, output)
    if kwinner2d_2_out_var.size()[0] == 0:
        kwinner2d_2_out_var = (output.data != 0).int()
    else:
        kwinner2d_2_out_var.add_((output.data != 0).int())

def print_linear_kwinner(self, input, output):
    global linear_kwinner_out_var
    global first_time
    if first_time:
        print_layer(self, input, output)
    if linear_kwinner_out_var.size()[0] == 0:
        linear_kwinner_out_var = (output.data != 0).int()
    else:
        linear_kwinner_out_var.add_((output.data != 0).int())

def count_mults(mat, vec):
    mmat = (mat != 0).int()
    vvec = (torch.t(vec) != 0).int()
    return(torch.matmul(mmat,vvec))

def print_sparse_linear_state(self, input, output):
    global sparse_linear_weights_var, sparse_linear_mults_var, sparse_weights, sparse_biases, sparse_instances, sparse_verify
    global first_time
    if first_time:
        print_layer(self, input, output)
        sparse_weights = self.module.weight.data
        sparse_biases = self.module.bias.data
        sparse_linear_weights_var = (self.module.weight.data != 0).int()
        sparse_linear_mults_var = count_mults(self.module.weight.data, input[0])
        sparse_instances = input[0]
        sparse_verify = output.data
    else:
        tmp = count_mults(self.module.weight.data, input[0])
        sparse_linear_mults_var = torch.cat((sparse_linear_mults_var, tmp), axis=1)
        sparse_instances = torch.cat((sparse_instances,input[0]), axis=0)
        sparse_verify = torch.cat((sparse_verify, output.data), axis=0)
    #print('InCount:',torch.nonzero(input[0]).size()[0], 'OutCount:', torch.nonzero(output.data).size()[0], 'WeightCount:',torch.nonzero(self.module.weight.data).size()[0])


"""
def print_linear_state(self, input, output):
    global linear_in_var
    global first_time
    if first_time:
        print_layer(self, input, output)

    if linear_in_var.size()[0] == 0:
        linear_in_var = (input[0] != 0).int()
    else:
        linear_in_var.add_((input[0] != 0).int())
    #print('InCount:', torch.nonzero(input[0]).size()[0], 'OutCount:', torch.nonzero(output.data).size()[0])

    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersparse", action="store_true", help="Use super sparse model instead of sparse model")
    #parser.add_argument("--pretrained", action="store_true",  help="Use pretrained model instead of training a new model")
    #parser.add_argument("--batchpow2min", default=1, type=int, help="Minimum power of 2 batch size (default 0)")
    #parser.add_argument("--batchpow2max", default=8, type=int, help="Maximum power of 2 batch size (default 8)")
    parser.add_argument("--batchsize", default=1, type=int,   help="Profile with a single batch size instead of sweeping powers of two")
    #parser.add_argument("--batchdelay", default=0, type=int, help="Sleep delay in seconds between batch inferences")
    #parser.add_argument("--numtrials", default=8, type=int, help="Number of outer loop iterations (default 8)")
    parser.add_argument("--datasetname", default="test", type=str, help='Dataset name (default = "test")')
    parser.add_argument("--datasetinflationfactor", default=1, type=int, help="Inflate dataset by this factor (default 1)")
    parser.add_argument("--dataloaderthreads", default=0, type=int, help="Number of subprocesses (threads) used by DataLoader (default 0)")

    args = parser.parse_args()

    processor_desc = get_processor_info()

    print("Processor Description:", processor_desc)
    print("Num torch threads:", torch.get_num_threads())
    print("Num physical cores:", psutil.cpu_count(False))
    print("Num logical cores:", psutil.cpu_count(True))
    print("Nominal CPU clock:", psutil.cpu_freq().current/1000.0, "GHz")

    datasetname = args.datasetname
    inflation_multiplier = args.datasetinflationfactor
    dataloader_threads = args.dataloaderthreads

    batch_size = args.batchsize
    # Use GPU if available
    devstring = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(devstring)
    modelclass = (gsc_super_sparse_cnn if args.supersparse else gsc_sparse_cnn)
    model = modelclass(True).to(device)
    print("Model:")
    print(model)

    global first_time
    first_time = True

    model.cnn1.register_forward_hook(print_layer)
    model.cnn1_batchnorm.register_forward_hook(print_layer)
    model.cnn1_maxpool.register_forward_hook(print_layer)

    global kwinner2d_1_out_var
    kwinner2d_1_out_var = torch.tensor([])
    model.cnn1_kwinner.register_forward_hook(print_kwinner2d_1) #kwinner2d 1

    model.cnn2.register_forward_hook(print_layer)
    model.cnn2_batchnorm.register_forward_hook(print_layer)
    model.cnn2_maxpool.register_forward_hook(print_layer)

    global kwinner2d_2_out_var
    kwinner2d_2_out_var = torch.tensor([])
    model.cnn2_kwinner.register_forward_hook(print_kwinner2d_2) #kwinner2d 2

    model.flatten.register_forward_hook(print_layer)

    global sparse_linear_weights_var, sparse_weights, sparse_biases, sparse_linear_mults_var, sparse_instances, sparse_verify
    sparse_linear_weights_var = torch.tensor([])
    sparse_weights = torch.tensor([]);
    sparse_biases = torch.tensor([])
    sparse_linear_mults_var = torch.tensor([])
    sparse_instances = torch.tensor([])
    sparse_verify = torch.tensor([])
    model.linear.register_forward_hook(print_sparse_linear_state)

    model.linear_bn.register_forward_hook(print_layer)

    global linear_kwinner_out_var
    linear_kwinner_out_var = torch.tensor([])
    model.linear_kwinner.register_forward_hook(print_linear_kwinner) #linear_kwinner

    #global linear_in_var
    #linear_in_var = torch.tensor([])
    #model.output.register_forward_hook(print_linear_state)
    model.output.register_forward_hook(print_layer)

    model.softmax.register_forward_hook(print_layer)


    """
    if not args.pretrained:
        cache_path = os.path.join("data", "cached_model.pth")

        # Option 1: Train model now
        do_training(model, device)
        torch.save(model.state_dict(), cache_path)

        # Option 2: Use previously saved model
        # model.load_state_dict(torch.load(cache_path))
    """
    fname = ('supersparse' if args.supersparse else 'sparse') + '_' + devstring + '_' + '"' + processor_desc + '"' + '_' + datetime.datetime.now().isoformat(sep='_', timespec='seconds') + '.csv'
    fname = fname.replace(':','-')
    wname = ('supersparse' if args.supersparse else 'sparse') + '_weights.bin'
    bname = ('supersparse' if args.supersparse else 'sparse') + '_biases.bin'
    iname = ('supersparse' if args.supersparse else 'sparse') + '_instances.bin'
    vname = ('supersparse' if args.supersparse else 'sparse') + '_verify.bin'
    with open(fname, 'x') as f:
        """"
        print("Processor Description:", processor_desc, file=f)
        print("Num torch threads:", torch.get_num_threads(), file=f)
        print("Num physical cores:", psutil.cpu_count(False), file=f)
        print("Num logical cores:", psutil.cpu_count(True), file=f)
        print("Nominal CPU clock:", psutil.cpu_freq().current / 1000.0, "GHz", file=f)

        print('Batch Size,Num Batches,Mean,Median,STDev,Min,Max', file=f)
        """
        res = do_profile_test(model, device, batch_size, datasetname, inflation_multiplier, dataloader_threads)
        #print(('{:4d},'*2 + '{:.3f},'*5).format(res[0],res[1],res[2],res[3],res[4],res[5],res[6]), file=f)
        with open(wname, "wb") as wf:
            sparse_weights.numpy().astype('float32').tofile(wf)
        with open(bname, "wb") as bf:
            sparse_biases.numpy().astype('float32').tofile(bf)
        with open(iname, "wb") as nf:
            sparse_instances.numpy().astype('float32').tofile(nf)
        with open(vname, "wb") as vf:
            sparse_verify.numpy().astype('float32').tofile(vf)
        print("Batch Size,", res[0],file=f)
        print("Num Batches,",res[1],file=f)
        print("kwin1_out,kwin2_out,kwinl_out,activations,sp_rows,sp_cols,mul_min,mul_max,mul_avg,mul_med,mul_std",file=f)
        ary2 = kwinner2d_1_out_var.numpy()[0].ravel()
        ary3 = kwinner2d_2_out_var.numpy()[0].ravel()
        ary1 = (sparse_instances.data != 0).int().numpy().sum(axis=1)
        mm = sparse_linear_mults_var.numpy()
        ary8 = mm.min(axis=1)
        ary9 = mm.max(axis=1)
        ary10 = mm.mean(axis=1)
        ary11 = np.median(mm, axis=1)
        ary12 = np.std(mm, axis=1)
        ary4 = sparse_linear_weights_var.numpy().sum(axis=0) # along col (across rows) (column non-sparsity)
        ary5 = sparse_linear_weights_var.numpy().sum(axis=1) # along row (across cols) (row non-sparsity)
        ary7 = linear_kwinner_out_var.numpy()[0]
        sz1 = ary1.shape[0]
        sz2 = ary2.shape[0]
        sz3 = ary3.shape[0]
        sz4 = ary4.shape[0]
        sz5 = ary5.shape[0]
        sz7 = ary7.shape[0]
        sz8 = ary8.shape[0]
        sz = max(sz2, sz3, sz4, sz5, sz7, sz8)

        for i in np.arange(sz):
            print("" if i >= sz2 else ary2[i],
                  "" if i >= sz3 else ary3[i],
                  "" if i >= sz7 else ary7[i],
                  "" if i >= sz1 else ary1[i],
                  "" if i >= sz5 else ary5[i],
                  "" if i >= sz4 else ary4[i],
                  "" if i >= sz8 else ary8[i],
                  "" if i >= sz8 else ary9[i],
                  "" if i >= sz8 else ary10[i],
                  "" if i >= sz8 else ary11[i],
                  "" if i >= sz8 else ary12[i],
                  sep=",", file=f)
        quit()
