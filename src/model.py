import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class TabularTimeSeriesModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=1e-3, model_type='mlp'):
        super(TabularTimeSeriesModel, self).__init__()
        self.save_hyperparameters()

        # Model selection logic
        if model_type == 'lstm':
            self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        elif model_type == 'bilstm':
            self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        elif model_type == 'mlp_lstm':  # MLP-LSTM directly embedded here
            self.fc1 = nn.Linear(15, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc5 = nn.Linear(32, output_size)
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(64)
            self.norm3 = nn.LayerNorm(32)

            self.lstm1 = nn.LSTM(6, 32, batch_first=True)
            self.lstm2 = nn.LSTM(32, 64, batch_first=True)
            self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        elif model_type == 'mlp':
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(8)]  # 3 hidden layers
            )
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(3)])
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(hidden_size, output_size)
        elif model_type == 'mlp_bilstm':
            # MLP + BiLSTM model definition
            self.fc1 = nn.Linear(15, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc5 = nn.Linear(32, 5)
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(64)
            self.norm3 = nn.LayerNorm(32)

            # BiLSTM layers
            self.lstm1 = nn.LSTM(6, 16, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
            self.lstm3 = nn.LSTM(64, 16, batch_first=True, bidirectional=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=output_size)
        self.val_f1 = MulticlassF1Score(num_classes=output_size,average='weighted')
        self.val_precision = MulticlassPrecision(num_classes=output_size,average='weighted')
        self.val_recall = MulticlassRecall(num_classes=output_size,average='weighted')
        self.test_acc = MulticlassAccuracy(num_classes=output_size)
        self.test_f1 = MulticlassF1Score(num_classes=output_size,average='weighted')
        self.test_precision = MulticlassPrecision(num_classes=output_size,average='weighted')
        self.test_recall = MulticlassRecall(num_classes=output_size,average='weighted')

        # Loss tracking for visualization
        self.training_losses = []
        self.validation_losses = []

    def forward(self, x):
        if self.hparams.model_type in ['lstm', 'bilstm']:
            if x.ndim == 2:
                x = x.unsqueeze(1)  # Add sequence dimension for LSTM/BiLSTM
            out, _ = self.model(x)
            out = self.fc(out[:, -1, :])  # Use the last hidden state
        elif self.hparams.model_type == 'mlp':
            x = torch.relu(self.fc1(x))
            for hidden_layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
                x = hidden_layer(x)
                x = batch_norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
            out = self.fc2(x)
        elif self.hparams.model_type == 'mlp_lstm':
            # Split the input for MLP and LSTM components
            x_MLP = x[:, :15]  # First 15 columns for MLP
            x_LSTM = x[:, 15:21].unsqueeze(1)  # Columns 16 to 33 for LSTM with sequence dim

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
        elif self.hparams.model_type == 'mlp_bilstm':
            # Split the input for MLP and BiLSTM components
            x_MLP = x[:, :15]  # First 15 columns for MLP
            x_LSTM = x[:, 15:21].unsqueeze(1)  # Columns 16 to 33 for BiLSTM (ensure sequence dimension)

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
        else:
            raise ValueError(f"Unsupported model type: {self.hparams.model_type}")
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.training_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate metrics
        f1 = self.test_f1(y_hat, y)
        precision = self.test_precision(y_hat, y)
        recall = self.test_recall(y_hat, y)
        acc = self.test_acc(y_hat, y)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_f1": f1, "test_precision": precision, "test_recall": recall, "test_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def visualize_training(self):
        # Ensure we plot only up to the actual number of epochs
        epochs = range(1, len(self.training_losses) + 1)  # Create a range for the x-axis
        # Visualize Training and Validation Losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()
