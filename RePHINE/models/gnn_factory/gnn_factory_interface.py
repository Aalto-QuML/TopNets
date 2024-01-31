from abc import ABC, abstractmethod


class GNNFactoryInterface(ABC):
    """"""

    @abstractmethod
    def return_gnn_instance(self, is_last=False):
        """"""
        pass
