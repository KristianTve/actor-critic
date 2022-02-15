import configparser


class config_manager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        # Initializing variables in config
        self.layers = None
        self.input_neurons = None
        self.output_neurons = None
        self.loss_func = None
        self.weight_reg = None
        self.weight_reg_rate = None
        self.min_weight = None
        self.max_weight = None
        self.activation_function = None
        self.softmax = None
        self.learning_rate = None
        self.batch_size = None
        self.verbose = None

        # Datagen config variables:
        self.images = None
        self.img_width = None
        self.img_height = None
        self.noise = None
        self.rand_pos = None
        self.train_pct = None
        self.test_pct = None
        self.val_pct = None

    def fetch_net_data(self):
        layer_size = []
        layer_act = []
        layer_min_weight = []
        layer_max_weight = []
        lr = []

        self.layers = self.config['GLOBALS']['Layers']
        self.input_neurons = self.config['GLOBALS']['Input_neurons']
        self.output_neurons = self.config['OUTPUT']['Output_neurons']
        self.loss_func = self.config['GLOBALS']['Loss_func']
        self.weight_reg = self.config['GLOBALS']['Weight_reg']
        self.weight_reg_rate = self.config['GLOBALS']['Weight_reg_rate']
        self.min_weight = self.config['GLOBALS']['Min_weight']
        self.max_weight = self.config['GLOBALS']['Max_weight']
        self.activation_function = self.config['GLOBALS']['Activation_function']
        self.softmax = self.config.getboolean('OUTPUT', 'Softmax')
        self.learning_rate = self.config['GLOBALS']['learning_rate']
        self.batch_size = self.config['GLOBALS']['batch_size']
        self.verbose = self.config.getboolean('GLOBALS', 'verbose')

        for i in range(int(self.layers)):
            layer_size.append(int(self.config['LAYER'+str(i+1)]['neurons']))
            layer_act.append(self.config['LAYER'+str(i+1)]['act'])
            layer_min_weight.append(float(self.config['LAYER'+str(i+1)]['min_weight']))
            layer_max_weight.append(float(self.config['LAYER'+str(i+1)]['max_weight']))
            lr.append(float(self.config['LAYER'+str(i+1)]['lr']))

        layer_min_weight.append(float(self.config['OUTPUT']['min_weight']))
        layer_max_weight.append(float(self.config['OUTPUT']['max_weight']))
        lr.append(float(self.config['OUTPUT']['lr']))
        layer_act.append(self.config['OUTPUT']['act'])
        return (int(self.layers),
                int(self.input_neurons),
                int(self.output_neurons),
                self.loss_func,
                self.weight_reg,
                float(self.weight_reg_rate),
                float(self.min_weight),
                float(self.max_weight),
                self.activation_function,
                bool(self.softmax),
                float(self.learning_rate),
                int(self.batch_size),
                bool(self.verbose),
                layer_size,
                layer_act,
                layer_min_weight,
                layer_max_weight,
                lr)

    def fetch_datagen_param(self):
        self.images = self.config['DATAGEN']['images']
        self.img_width = self.config['DATAGEN']['img_width']
        self.img_height = self.config['DATAGEN']['img_height']
        self.noise = self.config['DATAGEN']['noise']
        self.rand_pos = self.config['DATAGEN']['rand_pos']
        self.train_pct = self.config['DATAGEN']['train_pct']
        self.test_pct = self.config['DATAGEN']['test_pct']
        self.val_pct = self.config['DATAGEN']['val_pct']

        return (int(self.img_width),
                int(self.img_height),
                int(self.images),
                float(self.noise),
                float(self.train_pct),
                float(self.test_pct),
                float(self.val_pct),
                bool(self.rand_pos))
