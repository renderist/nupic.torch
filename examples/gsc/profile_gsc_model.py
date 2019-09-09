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


def train(model, loader, optimizer, criterion, device):
    """
    Train the model using given dataset loader.
    Called on every epoch.

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
    :type optimizer: :class:`torch.optim.Optimizer`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    """
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Train",
                                                    leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


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


def test(model, loader, criterion, device, desc="Test"):
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
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc=desc, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss += criterion(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}


def do_training(model, device):
    """
    Train the model.

    :param model: pytorch model to be trained
    :type model: torch.nn.Module

    :param device:
    :type device: torch.device
    """
    test_wavdata_to_tensor = [
        LoadAudio(),
        FixAudioLength(),
        ToMelSpectrogram(n_mels=32),
        ToTensor("mel_spectrogram", "input"),
        Unsqueeze("input"),
    ]

    valid_dataset = dataset_from_wavfiles(
        EXTRACTPATH/"valid", test_wavdata_to_tensor,
        cachefilepath=DATAPATH/"gsc_valid.npz")
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=VALID_BATCH_SIZE)

    test_dataset = dataset_from_wavfiles(
        EXTRACTPATH/"test", test_wavdata_to_tensor,
        cachefilepath=DATAPATH/"gsc_test.npz")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=TEST_BATCH_SIZE)

    train_wavdata_to_tensor = [
        LoadAudio(),
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        FixAudioLength(),
        ToSTFT(),
        StretchAudioOnSTFT(),
        TimeshiftAudioOnSTFT(),
        FixSTFTDimension(),
        ToMelSpectrogramFromSTFT(n_mels=32),
        DeleteSTFT(),
        ToTensor("mel_spectrogram", "input"),
        Unsqueeze("input"),
    ]
    sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1,
                                             gamma=LEARNING_RATE_GAMMA)
    for epoch in range(EPOCHS):
        train_dataset = dataset_from_wavfiles(
            EXTRACTPATH/"train", train_wavdata_to_tensor,
            cachefilepath=DATAPATH/"gsc_train{}.npz".format(epoch),
            silence_percentage=0.1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=(FIRST_EPOCH_BATCH_SIZE if epoch == 0
                        else TRAIN_BATCH_SIZE),
            shuffle=True)

        train(model=model, loader=train_loader, optimizer=sgd,
              criterion=F.nll_loss, device=device)
        if REDUCE_LR_ON_PLATEAU:
            validation = test(model=model, loader=valid_loader,
                              criterion=F.nll_loss, device=device,
                              desc="Validation")
            lr_scheduler.step(validation["loss"])
        else:
            lr_scheduler.step()
        model.apply(rezero_weights)
        model.apply(update_boost_strength)

        results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                       device=device)
        print("Epoch {}: {}".format(epoch, results))


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

def do_noise_test(model, device):
    """
    Test on the noisy data.

    :param model: pytorch model to be tested
    :type model: torch.nn.Module

    :param device:
    :type device: torch.device
    """
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        noise_wavdata_to_tensor = [LoadAudio(),
                                   FixAudioLength(),
                                   AddNoise(noise),
                                   ToMelSpectrogram(n_mels=32),
                                   ToTensor("mel_spectrogram", "input"),
                                   Unsqueeze("input")]
        cachefile = "gsc_test_noise{}.npz".format("{:.2f}".format(noise)[2:])
        test_dataset = dataset_from_wavfiles(EXTRACTPATH/"test",
                                             noise_wavdata_to_tensor,
                                             cachefilepath=DATAPATH/cachefile)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=TEST_BATCH_SIZE)
        results = test(model=model, loader=test_loader, criterion=F.nll_loss,
                       device=device)
        print("Noise level: {}, Results: {}".format(noise, results))


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supersparse", action="store_true", help="Use super sparse model instead of sparse model")
    parser.add_argument("--pretrained", action="store_true",  help="Use pretrained model instead of training a new model")
    parser.add_argument("--batchpow2min", default=1, type=int, help="Minimum power of 2 batch size (default 0)")
    parser.add_argument("--batchpow2max", default=8, type=int, help="Maximum power of 2 batch size (default 8)")
    parser.add_argument("--batchsize", default=0, type=int,   help="Profile with a single batch size instead of sweeping powers of two")
    parser.add_argument("--batchdelay", default=0, type=int, help="Sleep delay in seconds between batch inferences")
    parser.add_argument("--numtrials", default=8, type=int, help="Number of outer loop iterations (default 8)")
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

    numtrials = args.numtrials
    batchdelay = args.batchdelay
    batchpow2min = args.batchpow2min
    batchpow2max = args.batchpow2max
    datasetname = args.datasetname
    inflation_multiplier = args.datasetinflationfactor
    dataloader_threads = args.dataloaderthreads;


    # Use GPU if available
    devstring = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(devstring)
    modelclass = (gsc_super_sparse_cnn if args.supersparse else gsc_sparse_cnn)
    model = modelclass(pretrained=args.pretrained).to(device)
    #print("Model:")
    #print(model)

    if not args.pretrained:
        cache_path = os.path.join("data", "cached_model.pth")

        # Option 1: Train model now
        do_training(model, device)
        torch.save(model.state_dict(), cache_path)

        # Option 2: Use previously saved model
        # model.load_state_dict(torch.load(cache_path))

    #do_noise_test(model, device)
    fname = ('supersparse' if args.supersparse else 'sparse') + '_' + devstring + '_' + '"' + processor_desc + '"' + '_' + datetime.datetime.now().isoformat(sep='_', timespec='seconds') + '.csv'
    fname = fname.replace(':','-')
    if args.batchsize == 0:
        #batch_list = [1,2,4,8,16,32,64,128,256]
        #batch_list = [256,512,1024,2048,4096,8192]
        batch_list = [pow(2,x) for x in range(batchpow2min, batchpow2max+1)]
    else:
        batch_list = [args.batchsize,]
    with open(fname, 'x') as f:
        print("Processor Description:", processor_desc, file=f)
        print("Num torch threads:", torch.get_num_threads(), file=f)
        print("Num physical cores:", psutil.cpu_count(False), file=f)
        print("Num logical cores:", psutil.cpu_count(True), file=f)
        print("Nominal CPU clock:", psutil.cpu_freq().current / 1000.0, "GHz", file=f)

        print('Trial,Batch Size,Num Batches,Mean,Median,STDev,Min,Max', file=f)
        for trial in range(1,numtrials+1):
            print("Trial #", trial, sep="")
            for batch_size in batch_list:
                if batchdelay > 0:
                    time.sleep(batchdelay)
                res = do_profile_test(model, device, batch_size, datasetname, inflation_multiplier, dataloader_threads)
                res = [trial] + res
                print(('{:2d},'+'{:4d},'*2 + '{:.3f},'*5).format(res[0],res[1],res[2],res[3],res[4],res[5],res[6],res[7]), file=f)
