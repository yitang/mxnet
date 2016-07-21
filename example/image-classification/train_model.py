import find_mxnet
import mxnet as mx
import logging
import os

def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # devs = [mx.cpu(int(i)) for i in range(1)]
    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['accuracy', 'ce']
    ## TopKAccuracy only allows top_k > 1
    # for top_k in [5, 10, 20]:
    #     eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))
    
    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    #batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))
    batch_end_callback.append(log_train_eval_metric(5))

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)
from mxnet.callback import log_train_metric
def log_train_eval_metric(period, auto_reset=False):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.
    auto_reset : bool
        Reset the metric after each log

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0 and param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            for name, value in name_value:
                logging.info('Iter[%d] Batch[%d] Train-%s=%f',
                             param.epoch, param.nbatch, name, value)
            if auto_reset:
                param.eval_metric.reset()
    return _callback
