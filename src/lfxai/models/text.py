import logging
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


SOS_token = 2
EOS_token = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_length, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.hidden = None
        self.MAX_LEN = max_length
        self.device = device

    def forward(self, x):
        x.to(self.device)
        input_length = x.size(0)
        self.initHidden(self.device)
        outputs = torch.zeros(self.MAX_LEN, self.hidden_size, device=self.device)
        for ei in range(input_length):
            embedded = self.embedding(x[ei]).view(1, 1, -1)
            output = embedded
            output, self.hidden = self.gru(output, self.hidden)
            outputs[ei] = output[0,0]

        return self.hidden
        

    def initHidden(self, device):
        self.hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        return self.hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = None

    def forward(self, x):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, self.hidden = self.gru(output, self.hidden)
        output = self.softmax(self.out(output[0]))
        return output

    def initHidden(self, device):
        self.hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        return self.hidden


class AGNewsAE(nn.Module):
    def __init__(
        self,
        device: torch.device,
        max_length: int,
        vocab_size: int,
        embedding_dim: int = 64,
        name: str = "model",
        loss_f: callable = nn.NLLLoss(),
    ):
        super(AGNewsAE, self).__init__()
        self.encoder = EncoderRNN(vocab_size, embedding_dim, max_length, device)
        self.decoder = DecoderRNN(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.lr = None
        self.checkpoints_files = []
        self.name = name
        self.loss_f = loss_f
        self.teacher_forcing_ratio = 0.5
        self.MAX_LEN = max_length
        self.device = device


    def forward(self, x):

        target_length = x.size(0)

        encoder_output = self.encoder(x)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        self.decoder.hidden = encoder_output

        decoder_outputs = torch.zeros(target_length, self.vocab_size, device=self.device)

        for di in range(target_length):
            decoder_output = self.decoder(decoder_input)
            decoder_outputs[di] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() # detach from history as input

        return decoder_outputs

    
    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer
    ) -> np.ndarray:
        self.train()
        print_every =1000
        n_iters = len(dataloader)
        print_loss_total = 0
        n_iter = 0

        for input_tensor, _ in tqdm(dataloader, unit="batch", leave=False):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            target_length = input_tensor.size(0)
            target_tensor = input_tensor
            loss = 0

            encoder_output = self.encoder(input_tensor)

            decoder_input = torch.tensor([[SOS_token]], device=device)
            self.decoder.hidden = encoder_output

            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    target = torch.unsqueeze(target_tensor[di], 0)
                    decoder_output = self.decoder(decoder_input)
                    loss += self.loss_f(decoder_output, target)
                    decoder_input = target_tensor[di] # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    target = torch.unsqueeze(target_tensor[di], 0)
                    decoder_output = self.decoder(decoder_input)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach() # detach from history as input
                    loss += self.loss_f(decoder_output, target)
                    if decoder_input.item() == EOS_token:
                        break

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            print_loss_total += loss.item() / target_length

            if n_iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                logging.info(
                    f"iterations {n_iter}/{n_iters} \t "
                    f"loss {print_loss_avg:.3g} \t "
                )
            
            n_iter += 1

        return print_loss_avg


    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        n_iters = len(dataloader)
        loss_total = 0

        with torch.no_grad():
            for input_tensor, _ in tqdm(dataloader, unit="batch", leave=False):

                target_length = input_tensor.size(0)
                target_tensor = input_tensor
                loss = 0

                encoder_output = self.encoder(input_tensor)
                decoder_input = torch.tensor([[SOS_token]], device=device)
                self.decoder.hidden = encoder_output

                for di in range(target_length):
                    target = torch.unsqueeze(target_tensor[di], 0)
                    decoder_output = self.decoder(decoder_input)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach() # detach from history as input

                    loss += self.loss_f(decoder_output, target)
                    if decoder_input.item() == EOS_token:
                        break

                loss_total += loss.item() / target_length

        epoch_loss = loss_total / n_iters
        return epoch_loss

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        n_epoch: int = 5,
        checkpoint_interval: int = -1,
    ) -> None:

        self.to(device)
        self.lr = 1e-02
        best_test_loss = float("inf")

        encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.lr)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.lr)

        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, encoder_optimizer, decoder_optimizer)
            test_loss = self.test_epoch(device, test_loader)

            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss <= best_test_loss:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)


    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters:
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)