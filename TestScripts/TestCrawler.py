import glob , yaml, csv


class DataBuilder:
    def __init__(self, eval_name=None):
        self.data = {}
        self.base_keys = []
        self.n = 0
        self.path = f"Data/Vehicles/"
        self.eval_name = eval_name

        self.build_keys()
        self.read_data()
        self.save_data_table()

    def build_keys(self):
        with open(f"Data/base_key_builder.yaml") as f:
            key_data = yaml.safe_load(f)

        for key in key_data:
            self.base_keys.append(key)

    def read_data(self):
        store_n = 0
        reward_folders = glob.glob(f"{self.path}*/")
        for i, folder in enumerate(reward_folders):
            print(f"Folder being opened: {folder}")

            vehicle_folders = glob.glob(f"{folder}*/")
            for j, vehicle_folder in enumerate(vehicle_folders):
                print(f"Folder being opened: {vehicle_folder}")

                try:
                    config = glob.glob(vehicle_folder + '/*_record.yaml')[0]
                except Exception as e:
                    print(f"Exception: {e}")
                    print(f"Filename issue: {vehicle_folder}")
                    continue            

                with open(config, 'r') as f:
                    config_data = yaml.safe_load(f)

                if config_data is None:
                    continue

                self.data[store_n] = {}
                for key in config_data.keys():
                    if key in self.base_keys:
                        self.data[store_n][key] = config_data[key]
                store_n += 1

    def save_data_table(self, name="DataTable"):
        directory = "DataAnalysis/" + name + ".csv"
        with open(directory, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=self.base_keys)
            writer.writeheader()
            for key in self.data.keys():
            # for i in range(len(self.data.keys())):
                writer.writerow(self.data[key])


        print(f"Data saved to {name} --> {len(self.data)} Entries")


#TODO: in the future, read it in once and then just save it specifically according to eval

def run_builder():
    DataBuilder()


if __name__ == "__main__":
    run_builder()


