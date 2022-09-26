import PIL.Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


class SOCDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.csv_file_location = "data/trimmed.csv"
        self.work_csv_file_location = "data/work.csv"
        self.scaler = None
        self.df = pd.read_csv(self.csv_file_location)
        train, test = model_selection.train_test_split(self.df, test_size=0.2)
        self.df = train
        if not self.is_train:
            self.df = test

        self.df = self._preprocess(self.df)
        self.df.to_csv(self.work_csv_file_location, index = False)

    def _preprocess(self, df):
        self.__scale__(df)
        return df

    def __scale__(self, df):
        x = df[["soc"]].values.astype(float)
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(x)
        df["soc"] = x_scaled
        for col in df.columns[4:]:
            df = self.__scale_col__(df, col)
        return df

    def __scale_col__(self, df, col):
        x = df[[col]].values.astype(float)
        a_scaler = MinMaxScaler()
        x_scaled = a_scaler.fit_transform(x)
        df[col] = x_scaled
        return df

    def unscale(self, values):
        values = [[i] for i in values]
        values = self.scaler.inverse_transform(values)
        values = [i[0] for i in values]
        return values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        soc = row["soc"]
        x = list(row[4:])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)


if __name__ == "__main__":
    cid = SOCDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(soc)
        exit(0)

