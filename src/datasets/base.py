RAW_DATASET_ROOT_FOLDER = "data"

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from abc import *
from pathlib import Path
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, target_behavior, multi_behavior, min_uc):
        self.target_behavior = target_behavior
        self.multi_behavior = multi_behavior
        self.min_uc = min_uc  # 最小用户行为数
        self.bmap = None
        assert (
            self.min_uc >= 2
        ), "Need at least 2 items per user for validation and test"
        self.split = "leave_one_out"

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def load_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open("rb"))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print("Already preprocessed. Skip preprocessing")
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        df = self.load_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, umap, smap, bmap = self.densify_index(df)
        self.bmap = bmap
        train, train_b, val, val_b, val_num = self.split_df(df, len(umap))
        dataset = {
            "train": train,
            "val": val,
            "train_b": train_b,
            "val_b": val_b,
            "val_num": val_num,
            "umap": umap,
            "smap": smap,
            "bmap": bmap,
        }
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)

    def make_implicit(self, df):
        print("Behavior selection")
        if self.multi_behavior:
            pass
        else:
            # 只获取TargetBehavior的数据
            # 别的行为怎么处理的?
            # 嗷,是不是这样,代码这里根据multi_behavior来做一个类似于消融实验的处理,来看看不同行为的影响
            df = df[df["behavior"] == self.target_behavior]
        return df

    def filter_triplets(self, df):
        print("Filtering triplets")
        if self.min_uc > 0:
            # 过滤掉用户行为数小于min_uc的用户
            user_sizes = df.groupby("uid").size()  # 每一个用户对应的行为数
            good_users = user_sizes.index[
                user_sizes >= self.min_uc
            ]  # 行为数大于等于min_uc的用户
            df = df[df["uid"].isin(good_users)]  # 过滤掉行为数小于min_uc的用户
        return df

    def densify_index(self, df):
        print("Densifying index")
        # 为用户,商品,行为建立从1开始的索引
        umap = {u: (i + 1) for i, u in enumerate(set(df["uid"]))}
        smap = {s: (i + 1) for i, s in enumerate(set(df["sid"]))}
        bmap = {b: (i + 1) for i, b in enumerate(set(df["behavior"]))}
        # 将原始数据中的用户,商品,行为映射为新的索引
        df["uid"] = df["uid"].map(umap)
        df["sid"] = df["sid"].map(smap)
        df["behavior"] = df["behavior"].map(bmap)
        return df, umap, smap, bmap

    # def densify_index(self, df):
    #     print('Densifying index')
    #     umap = {u: u for u in set(df['uid'])}
    #     smap = {s: s for s in set(df['sid'])}
    #     bmap = {'pv': 1, 'fav':2, 'cart':3, 'buy':4} if 'buy' in set(df['behavior']) else {'tip': 1, 'neg':2, 'neutral':3, 'pos':4}
    #     df['behavior'] = df['behavior'].map(bmap)
    #     return df, umap, smap, bmap

    def split_df(self, df, user_count):
        if self.split == "leave_one_out":
            print("Splitting")
            user_group = df.groupby("uid")
            # since we have sorted raw input, we do not need to sort again,
            # if you use random permuted df, you need to use the following lines of code.
            # user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            # user2behaviors = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['behavior']))

            # 得到了每个用户的item序列
            user2items = user_group.progress_apply(lambda d: list(d["sid"]))
            # 得到了每个用户的behavior序列
            user2behaviors = user_group.progress_apply(lambda d: list(d["behavior"]))
            # split the dataset
            (
                train,
                train_b,
                val,
                val_b,
            ) = (
                {},
                {},
                {},
                {},
            )
            for user in range(1, user_count + 1):
                items = user2items[user]
                behaviors = user2behaviors[user]
                # only evaluate the target behavior
                if behaviors[-1] == self.bmap[self.target_behavior]:
                    train[user], val[user] = items[:-1], items[-1:]
                    train_b[user], val_b[user] = behaviors[:-1], behaviors[-1:]
                else:
                    train[user] = items
                    train_b[user] = behaviors
            return train, train_b, val, val_b, len(val)
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath("preprocessed")

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = "{}-min_uc{}-target_B{}_MB{}-split{}".format(
            self.code(),
            self.min_uc,
            self.target_behavior,
            self.multi_behavior,
            self.split,
        )
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath("dataset.pkl")
