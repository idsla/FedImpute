from abc import ABC, abstractmethod
import loguru
import timeit
import sys

from ..server import Server
from typing import Dict, Union, List, Tuple, Any
from ..client import Client
from ..utils.tracker import Tracker

class BaseWorkflow(ABC):

    """
    Abstract class for the workflow to be used in the federated imputation environment
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker
    ) -> Tracker:
        """
        Sequential federated imputation workflow

        Args:
            clients: List[Client] - list of clients
            server: Server - server
            evaluator: Evaluator - evaluator
            tracker: Tracker - tracker to tracking results

        Returns:
            Tracker - tracker with tracked results
        """
        pass

    @abstractmethod
    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker
    ) -> Tracker:
        """
        Parallel federated imputation workflow

        Args:
            clients: List[Client] - list of clients
            server: Server - server
            evaluator: Evaluator - evaluator
            tracker: Tracker - tracker to tracking results

        Returns:
            Tracker - tracker with tracked results
        """
        pass

    def run_fed_imp(
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker, run_type: str, verbose: int
    ) -> Tracker:

        """
        Run the federated imputation workflow based on the

        Args:
            clients: List[Client] - list of clients
            server:  Server - server
            evaluator: Evaluator - evaluator
            tracker: Tracker - tracker to tracking results
            run_type: str - type of the workflow run (sequential or parallel)

        Returns:
            Tracker - tracker with tracked results
        """
        # Set logging level based on verbosity
        loguru.logger.remove()  # Remove default handler
        if verbose == 0:
            pass
        elif verbose == 1:
            loguru.logger.add(
                sys.stdout, 
                level="SUCCESS",
                format="<level>{message}</level>"
            )
        elif verbose == 2:
            loguru.logger.add(
                sys.stdout, 
                level="INFO",
                format="<level>{message}</level>"
            ) 
        elif verbose >= 3:
            loguru.logger.add(
                sys.stdout, 
                level="DEBUG",
            )

        loguru.logger.success(f"Imputation Start ...")
        if run_type == 'sequential':
            start_time = timeit.default_timer()
            result = self.fed_imp_sequential(clients, server, evaluator, tracker)
            end_time = timeit.default_timer()
            loguru.logger.success(f"Finished. Running time: {end_time - start_time:.4f} seconds")
            return result
        elif run_type == 'parallel':
            start_time = timeit.default_timer()
            result = self.fed_imp_parallel(clients, server, evaluator, tracker)
            end_time = timeit.default_timer()
            loguru.logger.success(f"Finished. Running time: {end_time - start_time:.4f} seconds")
            return result
        else:
            raise ValueError('Invalid workflow run type')

    @staticmethod
    def eval_and_track(
        evaluator, 
        tracker, 
        clients, 
        phase='round', 
        epoch=0, 
        log_eval=True,
        log_metrics=None, 
        central_client=True, 
        other_infos=None,
        eval = True
    ) -> Union[None, Dict[str, Dict[str, float]]]:
        
        
        ############################################################################################################
        # Initial evaluation and tracking
        if phase == 'initial':
            
            if any(client.no_ground_truth for client in clients):
                
                clients_initial_stats = {
                    **(other_infos if other_infos is not None else {}),
                }
                
                avg_clients_stats = averaging_clients_stats(clients_initial_stats)
                
                tracker.record_initial(
                    data=[client.X_train for client in clients],
                    mask=[client.X_train_mask for client in clients],
                    imp_quality=None,
                )
                if log_eval:
                    loguru.logger.info("Initial Imputation.")
                
                return None
                
            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=[client.X_train_imp for client in clients],
                        X_train_origins=[client.X_train for client in clients],
                        X_train_masks=[client.X_train_mask for client in clients],
                        central_client=central_client
                    )
                    
                    clients_round_stats = {
                        **evaluation_results,
                        **(other_infos if other_infos is not None else {})
                    }
                    
                    avg_clients_stats = averaging_clients_stats(clients_round_stats)
                    
                    if log_eval:
                        if log_metrics is None or log_metrics not in avg_clients_stats:
                            logging_str = f"Initial: "
                            for key, value in avg_clients_stats.items():
                                logging_str += f"{key}: {value:.4f} "
                            loguru.logger.info(logging_str)
                        else:
                            logging_str = f"Initial: "
                            logging_str += f"{log_metrics}: {avg_clients_stats[log_metrics]:.4f}"
                            loguru.logger.info(logging_str)
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial Imputation."
                        )
                
                tracker.record_initial(
                    data=[client.X_train for client in clients],
                    mask=[client.X_train_mask for client in clients],
                    imp_quality=clients_round_stats,
                )

                return None

        ############################################################################################################
        # Evaluation and tracking for each round
        elif phase == 'round':
            
            if any(client.no_ground_truth for client in clients):
                
                # track other info
                tracker.record_round(
                    round_num=epoch + 1,
                    imp_quality=None,
                    data=[client.X_train_imp for client in clients],
                    model_params=[],  # todo make it
                    other_info=other_infos
                )
                
                clients_round_stats = {
                    **(other_infos if other_infos is not None else {}),
                }
                
                avg_clients_stats = averaging_clients_stats(clients_round_stats)
                
                if log_eval:
                    if log_metrics is None or log_metrics not in avg_clients_stats:
                        info_str = f"Epoch {epoch} "
                        for key, value in avg_clients_stats.items():
                            info_str += f"{key}: {value:.4f} "
                        loguru.logger.info(info_str)
                    else:
                        info_str = f"Epoch {epoch} "
                        info_str += f"{log_metrics}: {avg_clients_stats[log_metrics]:.4f}"
                        loguru.logger.info(info_str)
                
                return clients_round_stats
            
            else:
                if eval:
                    # evaluate imputation
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=[client.X_train_imp for client in clients],
                        X_train_origins=[client.X_train for client in clients],
                        X_train_masks=[client.X_train_mask for client in clients],
                        central_client=central_client,
                    )
                    
                    clients_round_stats = {
                        **evaluation_results,
                        **(other_infos if other_infos is not None else {})
                    }
                    
                    avg_clients_stats = averaging_clients_stats(clients_round_stats)
                    
                    # show evaluation results
                    if log_eval:
                        if log_metrics is None or log_metrics not in avg_clients_stats:
                            logging_str = f"Epoch {epoch}: "
                            for key, value in avg_clients_stats.items():
                                logging_str += f"{key}: {value:.4f} "
                            loguru.logger.info(logging_str)
                        else:
                            logging_str = f"Epoch {epoch}: "
                            logging_str += f"{log_metrics}: {avg_clients_stats[log_metrics]:.4f}"
                            loguru.logger.info(logging_str)
                        
                    # track other info and imputation quality
                    tracker.record_round(
                        round_num=epoch + 1,
                        imp_quality=evaluation_results,
                        data=[client.X_train_imp for client in clients],
                        model_params=[],  # todo make it
                        other_info=other_infos
                    )
                    
                    return clients_round_stats
                
                else:                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch} ..."
                        )
                        
                    clients_round_stats = {
                        **(other_infos if other_infos is not None else {}),
                    }
                    
                    avg_clients_stats = averaging_clients_stats(clients_round_stats)

                    # track other info and imputation quality
                    tracker.record_round(
                        round_num=epoch + 1,
                        imp_quality=None,
                        data=[client.X_train_imp for client in clients],
                        model_params=[],  # todo make it
                        other_info=other_infos
                    )

                    return clients_round_stats

        ############################################################################################################
        # Final evaluation and tracking
        elif phase == 'final':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_final(
                    imp_quality=None,
                    data=[client.X_train_imp for client in clients],
                    model_params=[],
                    other_info=None
                )
                
                clients_final_stats = {
                    **(other_infos if other_infos is not None else {}),
                }
                
                avg_clients_stats = averaging_clients_stats(clients_final_stats)
                
                if log_eval:
                    if log_metrics is None or log_metrics not in avg_clients_stats:
                        loguru.logger.info(f"Final Round ...")
                    else:
                        info_str = f"Final: "
                        info_str += f"{log_metrics}: {avg_clients_stats[log_metrics]:.4f}"
                        loguru.logger.info(info_str)
                
                return None
                
            else:
                
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=[client.X_train_imp for client in clients],
                        X_train_origins=[client.X_train for client in clients],
                        X_train_masks=[client.X_train_mask for client in clients],
                        central_client=central_client
                    )
                    
                    clients_final_stats = {
                        **evaluation_results,
                        **(other_infos if other_infos is not None else {})
                    }
                    
                    avg_clients_stats = averaging_clients_stats(clients_final_stats)
                    
                    if log_eval:
                        if log_metrics is None or log_metrics not in avg_clients_stats:
                            logging_str = f"Final: "
                            for key, value in avg_clients_stats.items():
                                logging_str += f"{key}: {value:.4f} "
                            loguru.logger.info(logging_str)
                        else:
                            logging_str = f"Final: "
                            logging_str += f"{log_metrics}: {avg_clients_stats[log_metrics]:.4f}"
                            loguru.logger.info(logging_str)
                    
                    tracker.record_final(
                        imp_quality=evaluation_results,
                        data=[client.X_train_imp for client in clients],
                        model_params=[],
                        other_info=None
                    )
                    
                    return clients_final_stats
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final Round ..."
                        )

                    tracker.record_final(
                        imp_quality=None,
                        data=[client.X_train_imp for client in clients],
                        model_params=[],
                        other_info=None
                    )
                
                    return None

    @staticmethod
    def eval_and_track_parallel(
        evaluator, tracker, clients_data, phase='round', epoch=0, log_eval=True, central_client=True, other_infos=None
    ):

        ############################################################################################################
        X_train_imps, X_train_origins, X_train_masks = [], [], []
        for X_train_imp, X_train_origin, X_train_mask in clients_data:
            X_train_imps.append(X_train_imp)
            X_train_origins.append(X_train_origin)
            X_train_masks.append(X_train_mask)

        ############################################################################################################
        # Initial evaluation and tracking
        if phase == 'initial':
            
            if any(client.no_ground_truth for client in clients):
                tracker.record_initial(
                    data=X_train_origins, mask=X_train_masks, imp_quality=None,
                )

                if log_eval:
                    loguru.logger.info("Initial Imputation.")
                
            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=X_train_imps, X_train_origins=X_train_origins, X_train_masks=X_train_masks,
                        central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial: "
                            f"rmse - {evaluator.get_average_imp_quality(evaluation_results, metric='rmse'):.4f} "
                            f"ws - {evaluator.get_average_imp_quality(evaluation_results, metric='ws'):.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial Imputation."
                        )

                tracker.record_initial(
                    data=X_train_origins, mask=X_train_masks, imp_quality=evaluation_results,
                )

            return None

        ############################################################################################################
        # Evaluation and tracking for each round
        elif phase == 'round':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_round(
                    round_num=epoch + 1,
                    imp_quality=None,
                    data=X_train_imps,
                    model_params=[],
                    other_info=other_infos
                )
                
                avg_other_info = {}
                for key, value in other_infos.items():
                    avg_other_info[key] = sum(value.values()) / len(value)
                
                info_str = f"Epoch {epoch} "
                for key, value in avg_other_info.items():
                    info_str += f"{key}: {value:.4f} "
                
                loguru.logger.info(info_str)
                
                return None

            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=X_train_imps, X_train_origins=X_train_origins, X_train_masks=X_train_masks,
                        central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch}: "
                            f"rmse - {evaluator.get_average_imp_quality(evaluation_results, metric='rmse'):.4f} "
                            f"ws - {evaluator.get_average_imp_quality(evaluation_results, metric='ws'):.4f}"
                        )
                    
                    tracker.record_round(
                        round_num=epoch + 1,
                        imp_quality=evaluation_results,
                        data=X_train_imps,
                        model_params=[],  # todo make it
                        other_info=other_infos
                    )
                     
                    return evaluation_results['imp_rmse']
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch} ..."
                        )

                    tracker.record_round(
                        round_num=epoch + 1,
                        imp_quality=None,
                        data=X_train_imps,
                        model_params=[],  # todo make it
                        other_info=other_infos
                    )

                    return None

        ############################################################################################################
        # Final evaluation and tracking
        elif phase == 'final':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_final(
                    imp_quality=None,
                    data=X_train_imps,
                    model_params=[],
                    other_info=None
                )

                if log_eval:
                    loguru.logger.info(f"Final Round ...")
                
                return None

            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=X_train_imps, X_train_origins=X_train_origins,
                        X_train_masks=X_train_masks, central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final: "
                            f"rmse - {evaluator.get_average_imp_quality(evaluation_results, metric='rmse'):.4f} "
                            f"ws - {evaluator.get_average_imp_quality(evaluation_results, metric='ws'):.4f}"
                        )
                    
                    tracker.record_final(
                        imp_quality=evaluation_results, data=X_train_imps, model_params=[],
                        other_info=None
                    )
                    
                    return evaluation_results['imp_rmse']
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final Round ..."
                        )

                    tracker.record_final(
                        imp_quality=None, 
                        data=X_train_imps, model_params=[],
                        other_info=None
                    )

                    return None


def averaging_clients_stats(clients_stats: dict) -> dict:
    avg_clients_stats = {}
    for key, value in clients_stats.items():
        avg_clients_stats[key] = sum(value.values()) / len(value)
    return avg_clients_stats


