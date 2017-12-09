import runmnist as rmn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,space_eval
from math import log


def objective(args):
	print(args)
	hyp = dict(
	    image_width = 28,
	    learning_rate = args['lr'],
	    reg_weight_inner = args['rwi'],
	    reg_weight_outer = args['rwo'],
	    output_bits = int(args['obits']),
	    layer_count = int(args['layers']),
	    qiter = 25,
	    iters = int(args['iter_count']),
	    batch = int(args['batch_size']),
	    quant_scheme = "partial_then_full",
	    quant_iter_threshold = args['qiter_threshold'], # switchover 75% of the way through
	    early_out = True,
	    partial_quant_threshold = args['partial_quant_threshold']
	  )
	accuracy,losslog,hist,uq_accuracy,q_accuracy = rmn.run_mnist(hyp)
	print("*********** Accuracy: " + str(accuracy))
	print(hyp)
	result = {
		'loss': -accuracy,
		'status': STATUS_OK,
		# -- store other results like this
		'losslog': losslog,
		'hist': hist,
		'uq_accuracy': uq_accuracy,
		'q_accuracy': q_accuracy,
		'accuracy': accuracy
		# -- attachments are handled differently
		#'attachments':
		#    {'time_module': pickle.dumps(time.time)}
	}
	return result


searchspace = {
    	'lr': hp.loguniform('lr',log(0.0001),log(0.1)),
    	'rwi': hp.loguniform('rwi',log(0.00000000001),log(0.0001)),
    	'rwo': hp.loguniform('rwo',log(0.01),log(10)),
    	'obits': hp.quniform('obits',6,16,1),
    	'layers': hp.quniform('layers',5,9,1),
    	'iter_count': hp.quniform('iter_count',300,700,25),
    	'batch_size': hp.quniform('batch_size',16,64,4),
    	'qiter_threshold': hp.uniform('qiter_threshold',0.7,0.8),
    	'partial_quant_threshold': 1.0 - hp.loguniform('om_partial_quant_threshold',log(0.01),log(1)),
}

trials = Trials()
best = fmin(objective,
    space=searchspace,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print(-best)
print(space_eval(space, best))
