

import logging


logging.basicConfig(
    format='%(asctime)s | %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news-vec.log'),
    ]
)

logger = logging.getLogger('news-vec')
