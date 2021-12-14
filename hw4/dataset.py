import torch
import h5py
from librosa.util import normalize


class LJSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, segment_len):
        super().__init__()

        self.data = h5py.File(data_path, 'r')
        self.keys = list(self.data.keys())
        self.segment_len = segment_len

    def __getitem__(self, index: int):
        item = self.data[str(index)]
        waveform = torch.from_numpy(normalize(item['wav'][:])) * 0.95

        if waveform.size(0) > self.segment_len:
            start = torch.randint(0, waveform.size(0) - self.segment_len)
            waveform = waveform[start:start + self.segment_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_len - waveform.size(0)))

        return waveform.unsqueeze(0)

    def __len__(self):
        return len(self.keys)