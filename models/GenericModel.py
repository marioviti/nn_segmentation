from keras.models import Model

class GenericModel(object):
    def __init__( self, inputs, outputs, loss, metrics, optimizer, loss_weights=None ):
        """
        params:
            inputs: (tuple)
            outputs: (tuple)
            loss:  (function) Optimization strategy.
            metrics: (tuple)
            optimizer: (optimizer)
        """
        self.model = Model(inputs=inputs, outputs=outputs)

        self.inputs_shape = [ input._keras_shape[1:] for input in inputs ]
        self.outputs_shape = [ output._keras_shape[1:] for output in outputs ]
        self.loss= loss
        self.loss_weights = loss_weights

        self.metrics = metrics
        self.optimizer = optimizer
        if loss_weights is None:
            self.model.compile( optimizer=self.optimizer,
                                loss=self.loss, metrics=self.metrics )
        else:
            self.model.compile( optimizer=self.optimizer,
                                loss_weights=self.loss_weights,
                                loss=self.loss, metrics=self.metrics )

    def save_model(self, name=None):
        self.name = self.name if name is None else name
        save_to( self.model,self.name )

    def load_model(self, name=None):
        self.name = self.name if name is None else name
        self.model = load_from( self.name )
        self.compile_model()

    def fit( self, x_train, y_train, batch_size=1, epochs=1, cropped=False ):
        return self.model.fit( x_train, y_train, \
                        epochs=epochs, batch_size=batch_size )

    def evaluate( self, x_test,  y_test,  batch_size=1, cropped=False ):
        return self.model.evaluate(x_test, y_test, batch_size=batch_size )

    def predict( self, x, batch_size=1, verbose=0 ):
        return self.model.predict( x, batch_size=batch_size, verbose=verbose )
