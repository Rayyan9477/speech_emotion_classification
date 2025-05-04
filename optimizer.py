import numpy as np
import random
from deap import base, creator, tools, algorithms
from model import EmotionModel
from trainer import ModelTrainer
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneticOptimizer:
    """
    Class for optimizing hyperparameters using genetic algorithms via DEAP.
    """
    def __init__(self, model_type='cnn', num_classes=7):
        """
        Initialize the GeneticOptimizer.
        
        Args:
            model_type (str): Type of model to optimize ('mlp' or 'cnn').
            num_classes (int): Number of emotion classes.
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.emotion_model = EmotionModel(num_classes=num_classes)
        
        # Set up genetic algorithm components
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def _setup_mlp_params(self):
        """
        Set up parameter ranges for MLP model optimization.
        
        Returns:
            tuple: (toolbox, parameter_ranges)
        """
        # Define parameter ranges
        param_ranges = {
            'learning_rate': (0.0001, 0.01),  # Continuous
            'num_layers': (1, 3),            # Discrete
            'units_layer1': (64, 256),       # Discrete
            'units_layer2': (32, 128),       # Discrete
            'units_layer3': (16, 64),        # Discrete
            'dropout_rate': (0.2, 0.5)       # Continuous
        }
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Register attribute generators
        toolbox.register("attr_learning_rate", random.uniform, param_ranges['learning_rate'][0], param_ranges['learning_rate'][1])
        toolbox.register("attr_num_layers", random.randint, param_ranges['num_layers'][0], param_ranges['num_layers'][1])
        toolbox.register("attr_units_layer1", random.randint, param_ranges['units_layer1'][0], param_ranges['units_layer1'][1])
        toolbox.register("attr_units_layer2", random.randint, param_ranges['units_layer2'][0], param_ranges['units_layer2'][1])
        toolbox.register("attr_units_layer3", random.randint, param_ranges['units_layer3'][0], param_ranges['units_layer3'][1])
        toolbox.register("attr_dropout_rate", random.uniform, param_ranges['dropout_rate'][0], param_ranges['dropout_rate'][1])
        
        # Register individual and population
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_learning_rate, toolbox.attr_num_layers, 
                         toolbox.attr_units_layer1, toolbox.attr_units_layer2, 
                         toolbox.attr_units_layer3, toolbox.attr_dropout_rate), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        return toolbox, param_ranges
    
    def _setup_cnn_params(self):
        """
        Set up parameter ranges for CNN model optimization.
        
        Returns:
            tuple: (toolbox, parameter_ranges)
        """
        # Define parameter ranges
        param_ranges = {
            'learning_rate': (0.0001, 0.01),  # Continuous
            'num_conv_layers': (1, 3),        # Discrete
            'filters_layer1': (16, 64),       # Discrete
            'filters_layer2': (32, 128),      # Discrete
            'filters_layer3': (64, 256),      # Discrete
            'num_dense_layers': (1, 2),       # Discrete
            'dense_units_layer1': (64, 256),  # Discrete
            'dense_units_layer2': (32, 128),  # Discrete
            'dropout_rate': (0.2, 0.5)        # Continuous
        }
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Register attribute generators
        toolbox.register("attr_learning_rate", random.uniform, param_ranges['learning_rate'][0], param_ranges['learning_rate'][1])
        toolbox.register("attr_num_conv_layers", random.randint, param_ranges['num_conv_layers'][0], param_ranges['num_conv_layers'][1])
        toolbox.register("attr_filters_layer1", random.randint, param_ranges['filters_layer1'][0], param_ranges['filters_layer1'][1])
        toolbox.register("attr_filters_layer2", random.randint, param_ranges['filters_layer2'][0], param_ranges['filters_layer2'][1])
        toolbox.register("attr_filters_layer3", random.randint, param_ranges['filters_layer3'][0], param_ranges['filters_layer3'][1])
        toolbox.register("attr_num_dense_layers", random.randint, param_ranges['num_dense_layers'][0], param_ranges['num_dense_layers'][1])
        toolbox.register("attr_dense_units_layer1", random.randint, param_ranges['dense_units_layer1'][0], param_ranges['dense_units_layer1'][1])
        toolbox.register("attr_dense_units_layer2", random.randint, param_ranges['dense_units_layer2'][0], param_ranges['dense_units_layer2'][1])
        toolbox.register("attr_dropout_rate", random.uniform, param_ranges['dropout_rate'][0], param_ranges['dropout_rate'][1])
        
        # Register individual and population
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_learning_rate, toolbox.attr_num_conv_layers, 
                         toolbox.attr_filters_layer1, toolbox.attr_filters_layer2, 
                         toolbox.attr_filters_layer3, toolbox.attr_num_dense_layers,
                         toolbox.attr_dense_units_layer1, toolbox.attr_dense_units_layer2,
                         toolbox.attr_dropout_rate), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        return toolbox, param_ranges
    
    def _evaluate_mlp(self, individual, X_train, y_train, X_val, y_val, input_shape):
        """
        Evaluate an individual (hyperparameter set) for MLP model.
        
        Args:
            individual (list): List of hyperparameters.
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation features.
            y_val (numpy.ndarray): Validation labels.
            input_shape (tuple): Shape of the input data.
            
        Returns:
            tuple: Tuple containing the fitness value (validation accuracy).
        """
        # Extract hyperparameters from individual
        learning_rate, num_layers, units_layer1, units_layer2, units_layer3, dropout_rate = individual
        
        # Round discrete parameters
        num_layers = int(num_layers)
        units_layer1 = int(units_layer1)
        units_layer2 = int(units_layer2)
        units_layer3 = int(units_layer3)
        
        # Create units list based on number of layers
        units = [units_layer1]
        if num_layers >= 2:
            units.append(units_layer2)
        if num_layers >= 3:
            units.append(units_layer3)
        
        # Create hyperparameter dictionary
        params = {
            'learning_rate': learning_rate,
            'num_layers': num_layers,
            'units': units,
            'dropout_rate': dropout_rate
        }
        
        try:
            # Clear Keras session to avoid memory issues
            tf.keras.backend.clear_session()
            
            # Build model with hyperparameters
            model = self.emotion_model.build_mlp(input_shape=input_shape, params=params)
            
            # Create trainer
            trainer = ModelTrainer(model, model_type='mlp')
            
            # Train model with early stopping
            callbacks = self.emotion_model.get_callbacks(patience=3)
            history = trainer.train(
                X_train, y_train,
                X_val, y_val,
                batch_size=32,
                epochs=20,  # Reduced epochs for optimization
                callbacks=callbacks
            )
            
            # Get validation accuracy from history
            val_accuracy = max(history.history['val_accuracy'])
            
            logger.info(f"MLP with params {params} achieved validation accuracy: {val_accuracy:.4f}")
            
            return (val_accuracy,)
        
        except Exception as e:
            logger.error(f"Error evaluating MLP individual: {e}")
            return (0.0,)  # Return low fitness for failed evaluations
    
    def _evaluate_cnn(self, individual, X_train, y_train, X_val, y_val, input_shape):
        """
        Evaluate an individual (hyperparameter set) for CNN model.
        
        Args:
            individual (list): List of hyperparameters.
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation features.
            y_val (numpy.ndarray): Validation labels.
            input_shape (tuple): Shape of the input data.
            
        Returns:
            tuple: Tuple containing the fitness value (validation accuracy).
        """
        # Extract hyperparameters from individual
        (learning_rate, num_conv_layers, filters_layer1, filters_layer2, filters_layer3,
         num_dense_layers, dense_units_layer1, dense_units_layer2, dropout_rate) = individual
        
        # Round discrete parameters
        num_conv_layers = int(num_conv_layers)
        filters_layer1 = int(filters_layer1)
        filters_layer2 = int(filters_layer2)
        filters_layer3 = int(filters_layer3)
        num_dense_layers = int(num_dense_layers)
        dense_units_layer1 = int(dense_units_layer1)
        dense_units_layer2 = int(dense_units_layer2)
        
        # Create filters list based on number of convolutional layers
        filters = [filters_layer1]
        if num_conv_layers >= 2:
            filters.append(filters_layer2)
        if num_conv_layers >= 3:
            filters.append(filters_layer3)
        
        # Create dense units list based on number of dense layers
        dense_units = [dense_units_layer1]
        if num_dense_layers >= 2:
            dense_units.append(dense_units_layer2)
        
        # Create hyperparameter dictionary
        params = {
            'learning_rate': learning_rate,
            'num_conv_layers': num_conv_layers,
            'filters': filters,
            'kernel_size': (3, 3),
            'pool_size': (2, 2),
            'num_dense_layers': num_dense_layers,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate
        }
        
        try:
            # Clear Keras session to avoid memory issues
            tf.keras.backend.clear_session()
            
            # Build model with hyperparameters
            model = self.emotion_model.build_cnn(input_shape=input_shape, params=params)
            
            # Create trainer
            trainer = ModelTrainer(model, model_type='cnn')
            
            # Train model with early stopping
            callbacks = self.emotion_model.get_callbacks(patience=3)
            history = trainer.train(
                X_train, y_train,
                X_val, y_val,
                batch_size=32,
                epochs=20,  # Reduced epochs for optimization
                callbacks=callbacks
            )
            
            # Get validation accuracy from history
            val_accuracy = max(history.history['val_accuracy'])
            
            logger.info(f"CNN with params {params} achieved validation accuracy: {val_accuracy:.4f}")
            
            return (val_accuracy,)
        
        except Exception as e:
            logger.error(f"Error evaluating CNN individual: {e}")
            return (0.0,)  # Return low fitness for failed evaluations
    
    def optimize(self, X_train, y_train, X_val, y_val, input_shape, 
                 population_size=10, generations=10, subset_size=None):
        """
        Optimize hyperparameters using genetic algorithm.
        
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation features.
            y_val (numpy.ndarray): Validation labels.
            input_shape (tuple): Shape of the input data.
            population_size (int): Size of the population.
            generations (int): Number of generations to evolve.
            subset_size (int): Size of the subset to use for evaluation (for efficiency).
            
        Returns:
            tuple: (best_individual, best_params, best_fitness)
        """
        # Use a subset of data for faster evaluation if specified
        if subset_size is not None and subset_size < len(X_train):
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
            
            val_indices = np.random.choice(len(X_val), subset_size // 5, replace=False)
            X_val_subset = X_val[val_indices]
            y_val_subset = y_val[val_indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
            X_val_subset = X_val
            y_val_subset = y_val
        
        logger.info(f"Starting hyperparameter optimization for {self.model_type.upper()} model")
        logger.info(f"Population size: {population_size}, Generations: {generations}")
        logger.info(f"Using {len(X_train_subset)} training samples and {len(X_val_subset)} validation samples")
        
        # Set up toolbox and parameter ranges based on model type
        if self.model_type == 'mlp':
            toolbox, param_ranges = self._setup_mlp_params()
            
            # Register evaluation function
            toolbox.register("evaluate", self._evaluate_mlp, 
                           X_train=X_train_subset, y_train=y_train_subset,
                           X_val=X_val_subset, y_val=y_val_subset,
                           input_shape=input_shape)
        else:  # CNN
            toolbox, param_ranges = self._setup_cnn_params()
            
            # Register evaluation function
            toolbox.register("evaluate", self._evaluate_cnn, 
                           X_train=X_train_subset, y_train=y_train_subset,
                           X_val=X_val_subset, y_val=y_val_subset,
                           input_shape=input_shape)
        
        # Register genetic operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        population = toolbox.population(n=population_size)
        
        # Track the best individual across all generations
        hof = tools.HallOfFame(1)
        
        # Track statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run the genetic algorithm
        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox,
                cxpb=0.7,  # Crossover probability
                mutpb=0.2,  # Mutation probability
                ngen=generations,
                stats=stats,
                halloffame=hof,
                verbose=True
            )
            
            # Get the best individual
            best_individual = hof[0]
            best_fitness = best_individual.fitness.values[0]
            
            # Convert best individual to parameter dictionary
            if self.model_type == 'mlp':
                learning_rate, num_layers, units_layer1, units_layer2, units_layer3, dropout_rate = best_individual
                
                # Round discrete parameters
                num_layers = int(num_layers)
                units_layer1 = int(units_layer1)
                units_layer2 = int(units_layer2)
                units_layer3 = int(units_layer3)
                
                # Create units list based on number of layers
                units = [units_layer1]
                if num_layers >= 2:
                    units.append(units_layer2)
                if num_layers >= 3:
                    units.append(units_layer3)
                
                best_params = {
                    'learning_rate': learning_rate,
                    'num_layers': num_layers,
                    'units': units,
                    'dropout_rate': dropout_rate
                }
            else:  # CNN
                (learning_rate, num_conv_layers, filters_layer1, filters_layer2, filters_layer3,
                 num_dense_layers, dense_units_layer1, dense_units_layer2, dropout_rate) = best_individual
                
                # Round discrete parameters
                num_conv_layers = int(num_conv_layers)
                filters_layer1 = int(filters_layer1)
                filters_layer2 = int(filters_layer2)
                filters_layer3 = int(filters_layer3)
                num_dense_layers = int(num_dense_layers)
                dense_units_layer1 = int(dense_units_layer1)
                dense_units_layer2 = int(dense_units_layer2)
                
                # Create filters list based on number of convolutional layers
                filters = [filters_layer1]
                if num_conv_layers >= 2:
                    filters.append(filters_layer2)
                if num_conv_layers >= 3:
                    filters.append(filters_layer3)
                
                # Create dense units list based on number of dense layers
                dense_units = [dense_units_layer1]
                if num_dense_layers >= 2:
                    dense_units.append(dense_units_layer2)
                
                best_params = {
                    'learning_rate': learning_rate,
                    'num_conv_layers': num_conv_layers,
                    'filters': filters,
                    'kernel_size': (3, 3),
                    'pool_size': (2, 2),
                    'num_dense_layers': num_dense_layers,
                    'dense_units': dense_units,
                    'dropout_rate': dropout_rate
                }
            
            logger.info(f"Optimization completed. Best fitness: {best_fitness:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return best_individual, best_params, best_fitness
        
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create optimizer
    optimizer = GeneticOptimizer(model_type='mlp', num_classes=7)
    
    # Generate dummy data for testing
    X_train = np.random.random((100, 13))
    y_train = np.random.randint(0, 7, size=(100,))
    X_val = np.random.random((20, 13))
    y_val = np.random.randint(0, 7, size=(20,))
    
    # Run optimization with small population and generations for testing
    best_individual, best_params, best_fitness = optimizer.optimize(
        X_train, y_train, X_val, y_val,
        input_shape=(13,),
        population_size=5,
        generations=3,
        subset_size=50
    )
    
    print(f"Best individual: {best_individual}")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness}")