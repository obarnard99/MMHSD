import yaml
import re


class Config:
    def __init__(self, config_path):
        with open(config_path) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
            p = re.compile('([A-Z])([0-9]+)([a-z]*)')
            m = p.match(self._config['model'])
            self._model, self._num_feats, flags = m.group(1), int(m.group(2)), m.group(3)
            if 'a' in flags:
                self._attr = True
            else:
                self._attr = False

            if self._model == 'U':
                self._num_pos = 6
                self._swa = False
            elif self._model == 'X':
                self._num_pos = 4
                self._swa = True
            elif self._model == 'O':
                self._num_pos = 4
                self._swa = False
            elif self._model == 'D':
                self._num_pos = 4
                self._swa = False

            if 'cased' in self._config['tr'].split('-'):
                self._case = True
            else:
                self._case = False

    @property
    def model(self):
        return self._model

    @property
    def num_feats(self):
        return self._num_feats

    @property
    def attr(self):
        return self._attr

    @property
    def seed(self):
        return self._config['seed']

    @property
    def num_pos(self):
        return self._num_pos

    @property
    def weights(self):
        return self._config['weights']

    @property
    def preweights(self):
        return self._config['preweights']

    @property
    def tr(self):
        return self._config['tr']

    @property
    def swa(self):
        return self._swa

    @property
    def case(self):
        return self._case
