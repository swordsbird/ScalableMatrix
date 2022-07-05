import sys
sys.path.append('..')
from lib.model_utils import ModelUtil
utils = ModelUtil(data_name = 'german', model_name = 'random forest')

# fuzzy coverage vector
print(utils.check_path(utils.paths[0], utils.X))

