from nvidia.dali.plugin.pytorch import DALIGenericIterator


class DALIGenericIteratorWithViews(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size=-1, reader_name=None):
        super().__init__(pipelines, output_map, size, reader_name=reader_name)

    def __next__(self):
        data = super().__next__()  # Get the batch of preprocessed images
        data = data[0]

        views = []
        for output_key in self.output_map:
            if "view" in output_key:
                views.append(data[output_key])
        return views, data["label"]

    def __len__(self):
        """Return the number of iterations per epoch."""
        # The following assumes there are more DALI pipelines. In our case there is only one
        samples_per_pipe = [sum(pipe.epoch_size().values()) for pipe in self._pipes]
        total_samples = sum(samples_per_pipe)
        return total_samples // self.batch_size
