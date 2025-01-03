from optimization.optimization_problem import (
    RegressionOptimizerProblem,
    ClassificationOptimizerProblem,
)
from mealpy.swarm_based.FFA import OriginalFFA
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.bio_based.SMA import OriginalSMA
from mealpy.math_based.SCA import OriginalSCA
from mealpy.human_based.QSA import ImprovedQSA
from utils.logger import setup_logger

logger = setup_logger(__name__)


def optimize(
    model_constructor,
    bounds,
    X,
    y,
    ml_model_name,
    metaheuristic,
    epoch,
    pop_size,
    filename,
    task_type,
):
    optimization_history = []

    if task_type == "classification":
        problem = ClassificationOptimizerProblem(
            bounds=bounds,
            minmax="min",
            X=X,
            y=y,
            ml_model_name=ml_model_name,
            ml_model_constructor=model_constructor,
        )
    else:
        problem = RegressionOptimizerProblem(
            bounds=bounds,
            minmax="min",
            X=X,
            y=y,
            ml_model_name=ml_model_name,
            ml_model_constructor=model_constructor,
        )

    if metaheuristic == "FFA":
        model = OriginalFFA(epoch=epoch, pop_size=pop_size)
    elif metaheuristic == "ABC":
        model = OriginalABC(epoch=epoch, pop_size=pop_size)
    elif metaheuristic == "HHO":
        model = OriginalHHO(epoch=epoch, pop_size=pop_size)
    elif metaheuristic == "SMA":
        model = OriginalSMA(epoch=epoch, pop_size=pop_size)
    elif metaheuristic == "SCA":
        model = OriginalSCA(epoch=epoch, pop_size=pop_size)
    elif metaheuristic == "ImprovedQSA":
        model = ImprovedQSA(epoch=epoch, pop_size=pop_size)
    else:
        raise ValueError(f"Unsupported metaheuristic algorithm: {metaheuristic}")

    model.solve(problem, seed=42, mode="thread", n_workers=20)
    # model.solve(problem, seed=42, mode="process", n_workers=80) #max n_workers=96

    if task_type == "classification":
        logger.info(
            f"Model: {ml_model_name}, Dataset: {filename}, Best Log Loss: {model.g_best.target.fitness}"
        )
    else:
        logger.info(
            f"Model: {ml_model_name}, Dataset: {filename}, Best MSE: {model.g_best.target.fitness}"
        )

    best_hyperparameters = model.problem.decode_solution(model.g_best.solution)
    optimized_model = model_constructor(**best_hyperparameters)

    optimization_history.append(
        [model.history.list_global_best_fit, model.history.list_epoch_time]
    )

    return optimized_model, best_hyperparameters, optimization_history
