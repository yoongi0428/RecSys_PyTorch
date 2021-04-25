import os
import sys
import warnings

from collections import OrderedDict
from configparser import ConfigParser

class Config:
    def __init__(self, main_conf_path):
        self.main_config = self.read_main_config(main_conf_path)

        exp_config = self.main_config['Experiment']
        self.model_config = self.read_model_config(exp_config['model_name'])

        self.read_system_arguments()

    def read_main_config(self, main_conf_path):
        main_config = OrderedDict()

        config = ConfigParser()
        config.read(main_conf_path)
        for section in config.sections():
            section_config = OrderedDict(config[section].items())
            main_config[section] = self.type_ensurance(section_config)

        return main_config

    def save_main_config(self, path):
        config_to_write = ConfigParser()
        config_to_write.read_dict(self.main_config)
        with open(path, 'w') as f:
            config_to_write.write(f)
        del config_to_write

    def read_model_config(self, model_name):
        model_config = OrderedDict()
        model_config_path = os.path.join('conf', model_name + '.cfg')

        config = ConfigParser()
        config.read(model_config_path)

        for section in config.sections():
            section_config = OrderedDict(config[section].items())
            model_config[section] = self.type_ensurance(section_config)

        return model_config
    
    def save_model_config(self, path):
        config_to_write = ConfigParser()
        config_to_write.read_dict(self.model_config)
        with open(path, 'w') as f:
            config_to_write.write(f)
        del config_to_write

    def read_system_arguments(self):
        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = dict()
            argvs = ' '.join(sys.argv[1:]).split(' ')
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i+1]
                arg_name = arg_name.strip('-')
                cmd_arg[arg_name] = arg_value
            self.update_params(self.type_ensurance(cmd_arg))

    def type_ensurance(self, config):
        BOOLEAN = {'true': True, 'false': False,
                   'True': True, 'False': False}

        for k, v in config.items():
            if isinstance(v, (int, float, list)):
                value = v
            else:
                try:
                    value = eval(v)
                    if not isinstance(value, (str, int, float, list, tuple)):
                        value = v
                except:
                    if v.lower() in BOOLEAN:
                        v = BOOLEAN[v.lower()]
                    value = v
            config[k] = value
        return config

    def get_param(self, section, param):
        if section in self.main_config:
            section = self.main_config[section]
        elif section in self.model_config:
            section = self.model_config[section]
        else:
            raise NameError("There are not the parameter named '%s'" % section)

        if param in section:
            value = section[param]
        else:
            raise NameError("There are not the parameter named '%s'" % param)

        return value

    def update_params(self, params):
        # for now, assume 'params' is dictionary

        params = self.type_ensurance(params)

        for k, v in params.items():
            updated=False
            for section in self.main_config:
                if k in self.main_config[section]:
                    self.main_config[section][k] = v
                    updated = True

                    if k == 'model_name':
                        self.model_config = self.read_model_config(v)
                    
                    break
            if not updated:
                for section in self.model_config:
                    if k in self.model_config[section]:
                        self.model_config[section][k] = v
                        updated = True
                        break

            if not updated:
                warnings.warn(f'{k}={v} is not recongnized on main/model config. \n Have no effect on Config.')

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("index must be a str")

        if item in self.main_config:
            section = self.main_config[item]
        elif item in self.model_config:
            section = self.model_config[item]
        else:
            raise NameError("There are not the parameter named '%s'" % item)
        return section

    def __str__(self):
        config_str = '\n'

        config_str += '>>>>> Main Config\n'
        for section in self.main_config:
            config_str += '[%s]\n' % section
            config_str += '\n'.join(['{}: {}'.format(k, self.main_config[section][k]) for k in self.main_config[section]])
            config_str += '\n\n'

        config_str += '>>>>> model Config\n'
        for section in self.model_config:
            config_str += '[%s]\n' % section
            config_str += '\n'.join(['{}: {}'.format(k, self.model_config[section][k]) for k in self.model_config[section]])
            config_str += '\n\n'

        return config_str

if __name__ == '__main__':
    param = Config('../main_config.cfg')

    print(param)
