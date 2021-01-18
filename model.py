#!/usr/bin/env python3


class FuseModel:
    def fuse(self, first, second):
        raise NotImplementedError()

    def train(self, train_args):
        raise NotImplementedError()


class LaserTagger(FuseModel):
    pass