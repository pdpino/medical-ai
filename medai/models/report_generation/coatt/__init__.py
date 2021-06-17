"""Re-implementation of the CoAtt paper.

- Code copied (or slightly adapted) from:
https://github.com/ZexinYan/Medical-Report-Generation
"""
import torch
from torch import nn
from torch.nn.functional import one_hot, pad

from medai.models.report_generation.coatt import att, cnn, lstm


class CoAttModel(nn.Module):
    def __init__(self,
                 vocab,
                 labels=range(14),
                 hidden_size=512,
                 embedding_size=512,
                 bn_momentum=0.1,
                 sentence_num_layers=1,
                 word_num_layers=1,
                 max_words=30,
                 # k=10,
                 imagenet=True,
                 cnn_model_name='resnet152',
                ):
        super().__init__()
        self.extractor = cnn.VisualFeatureExtractor(cnn_model_name, pretrained=imagenet)
        self.mlc = cnn.MLC(
            classes=labels,
            sementic_features_dim=hidden_size,
            fc_in_features=self.extractor.out_features,
        )
        self.co_attention = att.CoAttention(
            version='v4',
            embed_size=embedding_size,
            hidden_size=hidden_size,
            visual_size=self.extractor.out_features,
            # k=k, # Apparently is unused with v4
            momentum=bn_momentum,
        )

        self.sentence_model = lstm.SentenceLSTM(
            version='v1',
            embed_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=sentence_num_layers,
            dropout=0,
            momentum=bn_momentum,
        )
        self.word_model = lstm.WordLSTM(
            embed_size=embedding_size,
            hidden_size=hidden_size,
            vocab_size=len(vocab),
            num_layers=word_num_layers,
            n_max=max_words,
        )

        self.hidden_size = hidden_size
        self.vocab_size = len(vocab)

        # TODO: could the criterion be calculated outside forward() ??
        self.words_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.stops_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, images, captions, stops):
        # shapes:
        # images: bs, 3, 224, 224
        # captions: bs, n_sentences, n_words
        # stops: bs, n_sentences
        ### labels: bs, n_labels=210

        batch_size, n_sentences, n_words = captions.size()

        # Extract visual features
        _, avg_features = self.extractor(images)
        # shapes: (bs, n_features, 7, 7), (bs, n_features)

        # Classify tags
        tags, semantic_features = self.mlc(avg_features)
        # shapes: (bs, 156??), (bs, 10??, 512??)

        sentence_states = None
        hidden_states = torch.zeros(batch_size, 1, self.hidden_size, device=images.device)

        batch_word_loss = 0
        batch_stop_loss = 0

        # Collect words
        output_words = []
        output_stops = []

        for sentence_index in range(captions.size(1)):
            ctx, _, _ = self.co_attention(
                avg_features, semantic_features, hidden_states,
            )
            # shape ctx: bs, 512
            # shape others: (bs, 2048), (bs, 10, 512)

            topic, p_stop, hidden_states, sentence_states = self.sentence_model(
                ctx, hidden_states, sentence_states,
            )
            # shapes:
            # topic: bs, 1, hidden_size
            # p_stop: bs, 1, 2
            # hidden_states: bs, 1, hidden_size
            # sentence_states: tuple of two: (2, 1, 512) ??

            p_stop = p_stop.squeeze(1)
            # shape: bs, 2

            batch_stop_loss += self.stops_criterion(
                p_stop,
                stops[:, sentence_index],
            )

            # Save stops
            p_stop = p_stop.argmax(dim=-1) # shape: (bs,)
            output_stops.append(p_stop)

            if self.training:
                report_words = []
                for word_index in range(1, captions.size(2)):
                    word = self.word_model(
                        topic, captions[:, sentence_index, :word_index],
                    )
                    # shape: batch_size, vocab_size

                    batch_word_loss += self.words_criterion(
                        word, # shape: bs, vocab_size
                        captions[:, sentence_index, word_index], # shape: bs,
                    ) * (0.9 ** word_index)

                    report_words.append(word)

                report_words = torch.stack(report_words, dim=1)
                # shape: batch_size, n_words, vocab_size
            else:
                report_words = self.word_model(topic).long()
                # shape: batch_size, n_words

                report_words = one_hot(report_words, self.vocab_size)
                # shape: batch_size, n_words, vocab_size

                # Compute loss
                for word_index in range(1, captions.size(2)):
                    if word_index >= report_words.size(1):
                        break
                    word = report_words[:, word_index, :]
                    batch_word_loss += self.words_criterion(
                        word.float(), # shape: bs, vocab_size
                        captions[:, sentence_index, word_index], # shape: bs,
                    ) * (0.9 ** word_index)

            output_words.append(report_words)

        # Average losses
        batch_stop_loss /= n_sentences
        batch_word_loss /= (n_sentences * n_words)

        # if not isinstance(batch_word_loss, torch.Tensor):
        #     batch_word_loss = torch.zeros(1, device=images.device).fill_(-1)

        output_words = torch.stack(output_words, dim=1)
        # shape: batch_size, n_sentences, n_words, vocab_size

        output_stops = torch.stack(output_stops, dim=-1)
        # shape: batch_size, n_sentences

        return output_words, output_stops, tags, batch_stop_loss, batch_word_loss
