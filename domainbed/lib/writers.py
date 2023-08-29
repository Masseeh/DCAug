class Writer:
    def add_scalars(self, tag_scalar_dic, global_step):
        return

    def add_scalars_with_prefix(self, tag_scalar_dic, global_step, prefix):
        tag_scalar_dic = {prefix + k: v for k, v in tag_scalar_dic.items()}
        self.add_scalars(tag_scalar_dic, global_step)
    
    def close(self):
        return


class TBWriter(Writer):
    def __init__(self, dir_path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(dir_path, flush_secs=30)

    def add_scalars(self, tag_scalar_dic, global_step):
        for tag, scalar in tag_scalar_dic.items():
            self.writer.add_scalar(tag, scalar, global_step)
    
    def close(self):
        self.writer.close()


def get_writer(dir_path, tb):
    """
    Args:
        dir_path: tb dir
    """
    if tb:
        writer = TBWriter(dir_path)
    else:
        writer = Writer()

    return writer
