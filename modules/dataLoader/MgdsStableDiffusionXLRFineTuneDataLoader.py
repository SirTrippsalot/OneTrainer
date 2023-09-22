from modules.dataLoader.MgdsStableDiffusionXLRBaseDataLoader import MgdsStablDiffusionXLRBaseDataLoader
from modules.model.StableDiffusionXLRModel import StableDiffusionXLRModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsStableDiffusionXLRFineTuneDataLoader(MgdsStablDiffusionXLRBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionXLRModel,
            train_progress: TrainProgress,
    ):
        super(MgdsStableDiffusionXLRFineTuneDataLoader, self).__init__(args, model, train_progress)
