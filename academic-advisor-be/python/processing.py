import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy
import pytorch_lightning as pl
import matplotlib.pyplot as plt
#from torchxlstm import sLSTM, mLSTM, xLSTM
TABULAR_SIZE = 29
TABULAR_LSTM_SIZE = 46 #2nam cho 3 hoc ki
# TABULAR_LSTM_SIZE = 34 #2nam cho 3 hoc ki
#nam 1: 27
#nam 2: 32
#nam 3: 37
#3.5: 39
LSTM_SIZE = TABULAR_LSTM_SIZE - TABULAR_SIZE
MODEL_TYPE = 'lstm'
#mlp, mlp_lstm, mlp_bilstm,lstm, bilstm
EPOCHS = 20
empty_results_df = pd.DataFrame(columns=["Model","Accuracy", "F1", "Precision", "Recall"])
empty_results_df_v2 = pd.DataFrame(columns=["Model","Accuracy", "F1", "Precision", "Recall"])
class TabularTimeSeriesModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001, model_type='mlp',file_name = "default"):
        super(TabularTimeSeriesModel, self).__init__()
        self.save_hyperparameters()
        if model_type == 'lstm':
            print("lstm")
            self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        elif model_type == 'bilstm':
            print("bilstm")
            self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        elif model_type == 'mlp_lstm':  # MLP-LSTM directly embedded here
            print("mlp_lstm")
            self.fc1 = nn.Linear(TABULAR_SIZE, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc5 = nn.Linear(32, output_size)
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(64)
            self.norm3 = nn.LayerNorm(32)


            self.lstm1 = nn.LSTM(input_size, 32, batch_first=True)
            self.lstm2 = nn.LSTM(32, 64, batch_first=True)
            self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        elif model_type == 'mlp':
            print("mlp")
            self.fc1 = nn.Linear(TABULAR_SIZE, hidden_size)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(8)]  # 3 hidden layers
            )
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(3)])
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(hidden_size, output_size)
        elif model_type == 'mlp_bilstm':
            print("mlp_bilstm")
            # MLP + BiLSTM model definition
            self.fc1 = nn.Linear(TABULAR_SIZE, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc5 = nn.Linear(32, 5)
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(64)
            self.norm3 = nn.LayerNorm(32)


            # BiLSTM layers
            self.lstm1 = nn.LSTM(input_size, 16, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
            self.lstm3 = nn.LSTM(64, 16, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()


        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=output_size,average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=output_size,average='macro')
        self.val_precision = MulticlassPrecision(num_classes=output_size,average='macro')
        self.val_recall = MulticlassRecall(num_classes=output_size,average='macro')
        self.test_acc = MulticlassAccuracy(num_classes=output_size,average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=output_size,average='macro')
        self.test_precision = MulticlassPrecision(num_classes=output_size,average='macro')
        self.test_recall = MulticlassRecall(num_classes=output_size,average='macro')


        # Metrics
        self.test_acc_v2 = MulticlassAccuracy(num_classes=output_size,average='micro')
        self.test_f1_v2 = MulticlassF1Score(num_classes=output_size,average='weighted')
        self.test_precision_v2 = MulticlassPrecision(num_classes=output_size,average='weighted')
        self.test_recall_v2 = MulticlassRecall(num_classes=output_size,average='weighted')


        # Loss tracking for visualization
        self.training_losses = []
        self.validation_losses = []
        self.training_losses_avg = []
        self.validation_losses_avg = []


    def forward(self, x):
        if self.hparams.model_type in ['lstm', 'bilstm']:
            if x.ndim == 2:
                x_LSTM = x[:, TABULAR_SIZE:].unsqueeze(1)  # Add sequence dimension for LSTM/BiLSTM
            else:
                x_LSTM = x[:, TABULAR_SIZE:]  # If x is already 3D, just extract the relevant part


            out, _ = self.model(x_LSTM)  # Process with LSTM/BiLSTM
            out = self.fc(out[:, -1, :])  # Use the last hidden state for classification
            out = F.log_softmax(out, dim=1)  # Multi-class classification
        elif self.hparams.model_type == 'mlp':
            x_mlp = x[:, :TABULAR_SIZE]  # First part for MLP
            x = torch.relu(self.fc1(x_mlp))
            for hidden_layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
                x = hidden_layer(x)
                x = batch_norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
            out = self.fc2(x)
            out = F.log_softmax(out, dim=1)  # Multi-class classification
            return out
        elif self.hparams.model_type == 'mlp_lstm':
            # # Split the input for MLP and LSTM components
            x_MLP = x[:, :TABULAR_SIZE]  # First 15 columns for MLP
            x_LSTM = x[:, TABULAR_SIZE:].unsqueeze(1)  # Columns 16 to 33 for LSTM with sequence dim
            #print(x_LSTM.size())
            # LSTM layers
            x_LSTM1, _ = self.lstm1(x_LSTM)
            x_LSTM2, _ = self.lstm2(x_LSTM1)
            x_LSTM3, _ = self.lstm3(x_LSTM2)


            # MLP layers
            x_MLP = F.relu(self.norm1(self.fc1(x_MLP)))
            x_MLP = x_MLP + x_LSTM1.squeeze(1)  # Align dimensions after LSTM
            x_MLP = F.relu(self.norm2(self.fc2(x_MLP)))
            x_MLP = x_MLP + x_LSTM2.squeeze(1)  # Align dimensions after LSTM
            x_MLP = F.relu(self.norm3(self.fc3(x_MLP)))
            x_MLP = x_MLP + x_LSTM3.squeeze(1)  # Align dimensions after LSTM
            out = self.fc5(x_MLP)
            out = F.log_softmax(out, dim=1)  # Multi-class classification
            return out
        elif self.hparams.model_type == 'mlp_bilstm':
            # Split the input for MLP and BiLSTM components
            x_MLP = x[:, :TABULAR_SIZE]  # First 15 columns for MLP
            x_LSTM = x[:, TABULAR_SIZE:].unsqueeze(1)  # Columns 16 to 33 for BiLSTM (ensure sequence dimension)


            # BiLSTM layers
            x_LSTM1, _ = self.lstm1(x_LSTM)
            x_LSTM2, _ = self.lstm2(x_LSTM1)
            x_LSTM3, _ = self.lstm3(x_LSTM2)


            # MLP layers (feature extraction)
            x_MLP = F.relu(self.norm1(self.fc1(x_MLP)))
            x_MLP = x_MLP + x_LSTM1.squeeze(1)  # Combine MLP and BiLSTM output
            x_MLP = F.relu(self.norm2(self.fc2(x_MLP)))
            x_MLP = x_MLP + x_LSTM2.squeeze(1)  # Combine MLP and BiLSTM output
            x_MLP = F.relu(self.norm3(self.fc3(x_MLP)))
            x_MLP = x_MLP + x_LSTM3.squeeze(1)  # Combine MLP and BiLSTM output


            # Final output layer
            out = self.fc5(x_MLP)
            out = F.log_softmax(out, dim=1)  # Multi-class classification
            return out
        else:
            raise ValueError(f"Unsupported model type: {self.hparams.model_type}")
        return out


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        predicted_classes = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.training_losses.append(loss.item())
        return loss
    def on_train_epoch_end(self):
        print("trainning end")
        # Calculate the average training loss for the epoch
        avg_train_loss = sum(self.training_losses) / len(self.training_losses)
        self.log("avg_train_loss", avg_train_loss, prog_bar=True)
        self.training_losses_avg.append(avg_train_loss)
        # Clear the training_losses for the next epoch
        self.training_losses = []


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        predicted_classes = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)


        # Calculate metrics
        f1 = self.val_f1(y_hat, y)
        precision = self.val_precision(y_hat, y)
        recall = self.val_recall(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()


        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.validation_losses.append(loss.item())
        return {"val_loss": loss, "val_f1": f1, "val_precision": precision, "val_recall": recall, "val_acc": acc}
    def on_validation_epoch_end(self):
        print("validation end")
        # Calculate the average validation loss for the epoch
        avg_val_loss = sum(self.validation_losses) / len(self.validation_losses)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.validation_losses_avg.append(avg_val_loss)
        # Clear the validation_losses for the next epoch
        self.validation_losses = []


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # Get model predictions
        loss = self.loss_fn(y_hat, y)  # Compute loss


        # Store predictions and ground truth labels for later aggregation
        self.preds = self.preds if hasattr(self, 'preds') else []
        self.targets = self.targets if hasattr(self, 'targets') else []
        self.preds.append(y_hat)
        self.targets.append(y)


        # Return loss, which will be aggregated later
        return {"test_loss": loss}


    def on_test_epoch_end(self):
        print("test end")
        # Concatenate all the predictions and targets from test_step
        all_preds = torch.cat(self.preds, dim=0)
        predicted_classes = all_preds.argmax(dim=1)
        print(f'\npredicted class {predicted_classes}\n')
        print(f'\ntarget class {self.targets}\n')
        all_targets = torch.cat(self.targets, dim=0).view(-1)
        print(f'\ntarget class {self.targets}\n')
        # Get the length of the concatenated tensors
        num_samples = all_preds.size(0)


        print(f"Number of samples in predictions and targets: {num_samples}")
        # Calculate metrics on the entire dataset
        f1 = self.test_f1(all_preds, all_targets)
        precision = self.test_precision(all_preds, all_targets)
        recall = self.test_recall(all_preds, all_targets)
        acc = self.test_acc(all_preds, all_targets)


        f1_v2 = self.test_f1_v2(all_preds, all_targets)
        precision_v2 = self.test_precision_v2(all_preds, all_targets)
        recall_v2 = self.test_recall_v2(all_preds, all_targets)
        acc_v2 = self.test_acc_v2(all_preds, all_targets)
        # Aggregate loss (calculate the average of all losses)
        #avg_test_loss = torch.tensor([x["test_loss"] for x in self.trainer.test_loop.outputs]).mean().item()


        # Log the aggregated metrics
        #self.log("avg_test_loss", avg_test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)


        self.log("test_acc_v2", acc_v2, prog_bar=True)
        self.log("test_f1_v2", f1_v2, prog_bar=True)
        self.log("test_precision_v2", precision_v2, prog_bar=True)
        self.log("test_recall_v2", recall_v2, prog_bar=True)
        # Save metrics to a CSV file
        results = {
            "Model": [self.hparams.model_type],
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        }
        results_v2 = {
            "Model": [self.hparams.model_type],
            "Accuracy": acc_v2,
            "F1": f1_v2,
            "Precision": precision_v2,
            "Recall": recall_v2,
        }
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        results_df_v2 = pd.DataFrame(results_v2)
        # Append results to the global empty results DataFrame
        global empty_results_df
        global empty_results_df_v2
        empty_results_df = pd.concat([empty_results_df, results_df], ignore_index=True)
        empty_results_df_v2 = pd.concat([empty_results_df_v2, results_df_v2], ignore_index=True)


        # Optionally save the DataFrame to a CSV file
        #empty_results_df.to_csv("model_results.csv", index=False)
        # Print the final test metrics
        print(f" Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


    def visualize_training(self, train_losses, val_losses):
        epochs = range(1, len(train_losses) + 1)  # Create a range for the x-axis
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Average Training Loss")
        plt.plot(epochs, val_losses, label="Average Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()


base_columns = [
    "mssv", "diem_tt", "OTHER", "THPT", "ĐGNL", "hedt_CLC", "hedt_CNTN", "hedt_CQUI",
    "hedt_CTTT", "hedt_KSTN", "chuyennganh2_7340122_CLC", "chuyennganh2_7340122_CQ",
    "chuyennganh2_7480101_CLC", "chuyennganh2_7480101_CQ", "chuyennganh2_7480101_CTTN",
    "chuyennganh2_7480102_CLC", "chuyennganh2_7480102_CQ", "chuyennganh2_7480103_CLC",
    "chuyennganh2_7480103_CQ", "chuyennganh2_7480104_CLC", "chuyennganh2_7480104_CQ",
    "chuyennganh2_7480104_CTTT", "chuyennganh2_7480106_CLC", "chuyennganh2_7480106_CQ",
    "chuyennganh2_7480109_CQ", "chuyennganh2_7480201_CLC", "chuyennganh2_7480201_CQ",
    "chuyennganh2_7480202_CLC", "chuyennganh2_7480202_CQ", "chuyennganh2_7480202_CTTN",
]

selected_cols_1_years = ['sem1','sem2','sem3','term1','term2']#6
selected_cols_2_years = ['sem1','sem2','sem3','sem4','sem5','sem6','term1','term2','term3','term4']#11
selected_cols_3_years = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8','sem9','term1','term2','term3','term4','term5','term6']#16
selected_cols_3_5_years = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8','sem9','sem10','term1','term2','term3','term4','term5','term6','term7']#18

# selected_cols_1_years = ['sem1','sem2','sem3','term 1','term 2','label']#6
# selected_cols_2_years = ['sem1','sem2','sem3','sem4','sem5','sem6','term 1','term 2','term 3','term 4','label']#11
# selected_cols_3_years = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8','sem9','term 1','term 2','term 3','term 4','term 5','term 6','label']#16
# selected_cols_3_5_years = ['sem1','sem2','sem3','sem4','sem5','sem6','sem7','sem8','sem9','sem10','term 1','term 2','term 3','term 4','term 5','term 6','term 7','label']#18

oneyear =base_columns+selected_cols_1_years
twoyears = base_columns+selected_cols_2_years
threeyears =base_columns+selected_cols_3_years
three5years = base_columns+selected_cols_3_5_years

def process_file():
    dataset_test = pd.read_csv('../data_input/students_data.csv')

    major_columns = [
        "chuyennganh2_7340122_CLC", "chuyennganh2_7340122_CQ", "chuyennganh2_7480101_CLC",
        "chuyennganh2_7480101_CQ", "chuyennganh2_7480101_CTTN", "chuyennganh2_7480102_CLC",
        "chuyennganh2_7480102_CQ", "chuyennganh2_7480103_CLC", "chuyennganh2_7480103_CQ",
        "chuyennganh2_7480104_CLC", "chuyennganh2_7480104_CQ", "chuyennganh2_7480104_CTTT",
        "chuyennganh2_7480106_CLC", "chuyennganh2_7480106_CQ", "chuyennganh2_7480109_CQ",
        "chuyennganh2_7480201_CLC", "chuyennganh2_7480201_CQ", "chuyennganh2_7480202_CLC",
        "chuyennganh2_7480202_CQ", "chuyennganh2_7480202_CTTN"
    ]

    for col in major_columns:
        dataset_test[col] = 0

    for index, row in dataset_test.iterrows():
        major_code = row["majorCode"]
        if f"chuyennganh2_{major_code}" in major_columns:
            dataset_test.at[index, f"chuyennganh2_{major_code}"] = 1

    dataset_test.drop(columns=["majorCode"], inplace=True)

    education_columns = ["hedt_CLC", "hedt_CNTN", "hedt_CQUI", "hedt_CTTT", "hedt_KSTN"]

    for col in education_columns:
        dataset_test[col] = 0

    for index, row in dataset_test.iterrows():
        education_system = row["educationSystem"]
        if f"hedt_{education_system}" in education_columns:
            dataset_test.at[index, f"hedt_{education_system}"] = 1

    dataset_test.drop(columns=["educationSystem"], inplace=True)

    faculty_columns = ["khoa_CNPM", "khoa_HTTT", "khoa_KHMT", "khoa_KTMT", "khoa_KTTT", "khoa_MMT&TT"]
    for col in faculty_columns:
        dataset_test[col] = 0

    for index, row in dataset_test.iterrows():
        faculty = row["faculty"]
        if f"khoa_{faculty}" in faculty_columns:
            dataset_test.at[index, f"khoa_{faculty}"] = 1

    dataset_test.drop(columns=["faculty"], inplace=True)

    noisinh_groups = {
        "noisinh_0": ['Cộng hòa Séc', 'Liên Bang Nga', 'Australia', 'Campuchia'],
        "noisinh_1": ['Hà Giang', 'Cao Bằng', 'Lạng Sơn', 'Bắc Giang', 'Phú Thọ', 'Thái Nguyên', 'Bắc Kạn', 'Tuyên Quang', 'Lào Cai', 'Yên Bái', 'Lai Châu', 'Sơn La', 'Điện Biên', 'Hòa Bình'],
        "noisinh_2": ['Hà Nội', 'Hải Phòng', 'Hải Dương', 'Hưng Yên', 'Vĩnh Phúc', 'Bắc Ninh', 'Thái Bình', 'Nam Định', 'Hà Nam', 'Ninh Bình', 'Quảng Ninh'],
        "noisinh_3": ['Thanh Hóa', 'Nghệ An', 'Hà Tĩnh', 'Quảng Bình', 'Quảng Trị', 'Thừa Thiên - Huế', 'Đà Nẵng', 'Quảng Nam', 'Quảng Ngãi', 'Bình Định', 'Phú Yên', 'Khánh Hòa', 'Ninh Thuận', 'Bình Thuận'],
        "noisinh_4": ['Kon Tum', 'Gia Lai', 'Đắk Lắk', 'Đắk Nông', 'Lâm Đồng'],
        "noisinh_5": ['TP. Hồ Chí Minh', 'Đồng Nai', 'Bà Rịa-Vũng Tàu', 'Bình Dương', 'Bình Phước', 'Tây Ninh'],
        "noisinh_6": ['Cần Thơ', 'Long An', 'Tiền Giang', 'Bến Tre', 'Trà Vinh', 'Vĩnh Long', 'An Giang', 'Đồng Tháp', 'Kiên Giang', 'Hậu Giang', 'Sóc Trăng', 'Bạc Liêu', 'Cà Mau']
    }

    for col in noisinh_groups.keys():
        dataset_test[col] = 0

    for index, row in dataset_test.iterrows():
        place = row["placeOfBirth"]
        for col, locations in noisinh_groups.items():
            if place in locations:
                dataset_test.at[index, col] = 1
                break  

    dataset_test.drop(columns=["placeOfBirth"], inplace=True)
    new_columns = major_columns + education_columns + faculty_columns + list(noisinh_groups.keys())

    all_columns = list(dataset_test.columns)

    dgnl_index = all_columns.index("ĐGNL")
    sem1_index = all_columns.index("sem1")

    columns_before_dgnl = all_columns[:dgnl_index + 1]  # Cột trước và bao gồm ĐGNL
    columns_after_sem1 = all_columns[sem1_index:]  # Cột từ sem1 trở đi
    reordered_columns = columns_before_dgnl + new_columns + columns_after_sem1

    dataset_test = dataset_test[reordered_columns]

    term16_index = dataset_test.columns.get_loc("term16")
    columns_to_keep = dataset_test.columns[:term16_index + 1]
    dataset_test = dataset_test[columns_to_keep]

    dataset_test = dataset_test.loc[:, ~dataset_test.columns.duplicated()]

    # dataset_test.to_csv('../output/output_file.csv', index=False)

    # dataset = dataset[three5years]
    dataset_test = dataset_test[three5years]

    model = TabularTimeSeriesModel.load_from_checkpoint('../model.ckpt')
    model.freeze()

    #prepare data for testing
    x_test = dataset_test.drop(columns=['mssv'])
    # x_test = dataset.drop(columns=['mssv', 'label'])
    #x_test = dataset_test[oneyear].values
    x_test = x_test.iloc[:, :TABULAR_LSTM_SIZE].values  # Select the first 21 columns after dropping

    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)

    predicted_class = torch.argmax(y_pred, dim=1)
    output_file = "../output/predicted_classes.txt"

    with open(output_file, "w") as file:
        for pred in predicted_class:
            file.write(f"{pred.item()}\n")

    print(f"Kết quả đã được ghi vào tệp: {output_file}")

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("students_data.csv"):
            process_file()

# Lắng nghe thay đổi của file
if __name__ == "__main__":
    path = "../data_input/"
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print("Đang lắng nghe thay đổi file...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()