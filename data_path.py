"""
This file records the directory paths to the different datasets.
You will need to configure it for training the model.

All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

DATA_PATH = {
    'videomatte240k': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        }
    },
    'photomatte13k': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        }
    },
    'distinction': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'adobe': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'backgrounds': {
        'train': 'Backgrounds_Validation',
        'valid': 'Backgrounds_Validation'
    },
    'sprite-dx-data': {
        'train': {
            'fgr': '../sprite-dx-data/data/fgr/random',
            'pha': '../sprite-dx-data/data/automatte/random'
        },
        'valid': {
            'fgr': '../sprite-dx-data/data/fgr/sample-100',
            'pha': '../sprite-dx-data/data/automatte/sample-100'
        }
    },
    'sprite-dx-all': {
        'train': {
            'fgr': '../sprite-dx-data/data/fgr',
            'pha': '../sprite-dx-data/data/automatte'
        },
        'valid': {
            'fgr': '../sprite-dx-data/data/fgr/sample-100',
            'pha': '../sprite-dx-data/data/automatte/sample-100'
        }
    },
    'sprite-dx-100': {
        'train': {
            'fgr': '../sprite-dx-data/data/fgr/sample-100',
            'pha': '../sprite-dx-data/data/automatte/sample-100'
        },
        'valid': {
            'fgr': '../sprite-dx-data/data/fgr/sample-100',
            'pha': '../sprite-dx-data/data/automatte/sample-100'
        }
    },
    'sprite-dx-193': {
        'train': {
            'fgr': '../sprite-dx-data/data/fgr/sample-193',
            'pha': '../sprite-dx-data/data/automatte/sample-193'
        },
        'valid': {
            'fgr': '../sprite-dx-data/data/fgr/sample-193',
            'pha': '../sprite-dx-data/data/automatte/sample-193'
        }
    },
    'sprite-dx-045': {
        'train': {
            'fgr': '../sprite-dx-data/data/fgr/sample-045',
            'pha': '../sprite-dx-data/data/automatte/sample-045'
        },
        'valid': {
            'fgr': '../sprite-dx-data/data/fgr/sample-045',
            'pha': '../sprite-dx-data/data/automatte/sample-045'
        }
    },
    'sprite-dx-000': {
        'train': {
            'fgr': '../sprite-dx-data/data/fgr/sample-000',
            'pha': '../sprite-dx-data/data/automatte/sample-000'
        },
        'valid': {
            'fgr': '../sprite-dx-data/data/fgr/sample-000',
            'pha': '../sprite-dx-data/data/automatte/sample-000'
        }
    },
}