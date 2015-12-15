import os
import sys
import multiprocessing as mp
import logging

from .config import Config

logger = logging.getLogger('hexrd.config')

class MultiprocessingConfig(Config):

    @property
    def ncpus(self):
        multiproc = self._cfg.get('multiprocessing:ncpus', default=-1)
        ncpus = mp.cpu_count()
        if multiproc == 'all':
            res = ncpus
        elif multiproc == 'half':
            temp = ncpus / 2
            res = temp if temp else 1
        elif isinstance(multiproc, int):
            if multiproc >= 0:
                if multiproc > ncpus:
                    logger.warning(
                        'Resuested %s processes, %d available',
                        multiproc, ncpus, ncpus
                        )
                    res = ncpus
                else:
                    res = multiproc if multiproc else 1
            else:
                temp = ncpus + multiproc
                if temp < 1:
                    logger.warning(
                        'Cannot use less than 1 process, requested %d of %d',
                        temp, ncpus
                        )
                    res = 1
                else:
                    res = temp
        else:
            temp = ncpus - 1
            logger.warning(
                "Invalid value %s (type %s) for multiprocessing",
                multiproc, type(multiproc)
                )
            res = temp
        return res

    @property
    def chunksize(self):
        res = self._cfg.get('multiprocessing:chunksize', 1000)
        if res == 'max':
            res = sys.maxint
        elif not isinstance(res, int) or res <= 0:
            logger.warning(
                "Invalid chunk size %s",
                str(res)
                )
            res = sys.maxint
        return res
