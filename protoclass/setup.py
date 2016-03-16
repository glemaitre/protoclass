def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('protoclass', parent_package, top_path)

    config.add_subpackage('classification')
    config.add_subpackage('extraction')
    config.add_subpackage('preprocessing')
    config.add_subpackage('selection')
    config.add_subpackage('tool')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')
    config.add_subpackage('validation')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
