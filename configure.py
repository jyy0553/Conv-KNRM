from tensorflow import flags

flags.DEFINE_integer("embedding_dim",50, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_string("filter_sizes", "20", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
# flags.DEFINE_float("dropout_keep_prob",0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")
flags.DEFINE_float("learning_rate", 0.0001, "learn rate( default: 0.0)")
flags.DEFINE_integer("max_len_query", 4, "max document length of left input")
flags.DEFINE_integer("max_len_document", 50, "max document length of right input")

flags.DEFINE_string("loss","point_wise","loss function (default:point_wise)")

flags.DEFINE_string('data','clueweb09B-title','the name of dataset')
# flags.DEFINE_string("dataset","clueweb09B-title","the name of dataset")

flags.DEFINE_string("file_name","File1","cross_validation file name")

flags.DEFINE_integer('extend_feature_dim',10,'overlap_feature_dim')

flags.DEFINE_integer("kener_num",11,"the num of kener")

# Training parameters
flags.DEFINE_integer("batch_size", 8, "Batch Size (default: 64)")
flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
flags.DEFINE_integer("num_epochs", 5000, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
flags.DEFINE_boolean('overlap_needed',True,"is overlap used")
flags.DEFINE_boolean('position_needed',False,'is position used')
flags.DEFINE_boolean('dns','False','whether use dns or not')
flags.DEFINE_string('CNN_type','ircnn','data set')
flags.DEFINE_float('sample_train',1,'sampe my train data')
flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
flags.DEFINE_string('pooling','max','pooling strategy')
flags.DEFINE_boolean('clean',False,'whether clean the data')

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#data_helper para

flags.DEFINE_boolean('isEnglish',True,'whether data is english')
