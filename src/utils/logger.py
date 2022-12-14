import torch


class Logger(object):
    def __init__(self, output_dir: str, filename: str):
        self.output_file = f"{output_dir}/{filename}_log.txt"
        self.results = []

    def add_result(self, result):
        assert len(result) == 3
        self.results.append(result)

    def save_statistics(self):
        result = 100 * torch.tensor(self.results)

        with open(self.output_file, "w") as txt_file:
            txt_file.write("Train, Validation, Final Train, Test\n")
            for r in result:
                train1 = r[0].item()
                valid = r[1].item()
                test = r[2].item()
                txt_file.write("{},{},{}\n".format(train1, valid, test))
