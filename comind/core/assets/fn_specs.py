from comind.llm.llm import FunctionSpec

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (4-5 sentences) describing the empirical findings. "
                "Examine wheter its goals are achieved. "
                "Summarize the output and mention if the submission.csv was properly produced. "
                "DO NOT suggest fixes or improvements.",
            },
            "output_abs": {
                "type": "string",
                "description": "select representative segments of the output log and mark the remainder as ellipses."
            },
            "metric": {
                "type": "number",
                "description": "If the code **ran successfully and produced submission.csv on full test set (i.e. not dummy or partial)**, report the value of the **final** validation metric. Otherwise, do not provide this field."
            },
            "is_lower_better": {
                "type": "boolean", 
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy). If the code terminated before producing the submission.csv, do not provide this field."
            }
        },
        "required": [
            "is_bug",
            "summary",
            "output_abs"
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

propose_code_func_spec = FunctionSpec(
    name="propose_code",
    json_schema={
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Describe your plan and what the code you are about to write will do. Explain your reasoning for the proposed code.",
            },
            "code": {
                "type": "string",
                "description": "The complete Python code to be executed to solve the task or the next step.",
            },
        },
        "required": ["explanation", "code"],
    },
    description="Propose Python code to be executed in the environment to solve a data science task.",
)

report_func_spec = FunctionSpec(
    name="submit_report",
    json_schema={
        "type": "object",
        "properties": {
            "pipeline": {
                "type": "string",
                "description": "A detailed description of the pipeline that generated the best results. All hyperparameters, training settings, model architectures, feature engineering, validation metric, and any other relevant information should be included. Describe potential improvements and future work. ",
            },
            "summary": {
                "type": "string",
                "description": (
                    "A comprehensive evaluation of each individual component of the pipeline. For each component, summarize in the following format: \n"
                    "- The component name\n"
                    "    Novelty: 0-10 (0: trivial, 10: clearly novel - major differences from existing well-known methods)\n"
                    "    Your Rationale \n\n"
                    "    Feasibility: 0-10 (0: almost impossible to implement and require extensive engineering, 10: Easy to implement)\n"
                    "    Your Rationale \n\n"
                    "    Effectiveness: 0-10 (0: minimal performance improvement, 10: very strong performance, significantly outperform most baselines)\n"
                    "    Your Rationale \n\n"
                    "    Efficiency: 0-10 (0: very slow, over-dependent on CPU and hard to produce meaningful results within the time limit, 10: high utilization of GPU)\n"
                    "    Your Rationale \n\n"
                    "    Confidence: 0-10 (0: no emprical results, not sure whether the evaluation is correct, 10: fully verified on large scale with abundant results)\n"
                )
            },
            "suggestions": {
                "type": "string",
                "description": (
                    "An extremely detailed description of the weaknesses of the pipeline you found during the implementation. Determine the bottleneck of the pipeline and suggest possible improvements. "
                )
            }
        },
        "required": [
            "pipeline",
            "summary",
            "suggestions",
        ],
    },
    description="Submit a comprehensive report evaluating the whole pipeline.",
)

propose_idea_func_spec = FunctionSpec(
    name="propose_idea",
    json_schema={
        "type": "object",
        "properties": {
            "ideas": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "A description of an idea. You should keep the idea self-contained and not overlap with the existing ideas and pipelines."
                },
            },
        },
        "required": [
            "ideas",
        ],
    },
    description="Propose ideas that haven't been covered yet."
)

submit_pipeline_func_spec = FunctionSpec(
    name="submit_pipeline", 
    json_schema={
        "type": "object",
        "properties": {
            "pipelines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pipeline": {
                            "type": "string",
                            "description": "A full detailed description of the pipeline, all input/output format, hyperparameters, training settings, model architectures, feature engineering, validation metric, and any other relevant information should be included. **Do not omit any feature engineering details**"
                        },
                        "code_abs": {
                            "type": "string", 
                            "description": "A representative code segments that captures the essence (including input/output) and novelty of the pipeline. You **MUST** go through all the publicly available code and **include the parts that generate the submission file**. Contain task-specific engineering details. Mark the remainder as ellipses."
                        }
                    },
                    "required": ["pipeline", "code_abs"]
                }
            }
        },
        "required": ["pipelines"],
    },
    description="Submit a list of pipelines."
)

metric_direction_func_spec = FunctionSpec(
    name="determine_metric_direction",
    json_schema={
        "type": "object",
        "properties": {
            "is_lower_better": {
                "type": "boolean",
                "description": "True if lower values are better (e.g., MSE, RMSE, MAE, cross-entropy loss), False if higher values are better (e.g., accuracy, F1-score, AUC, precision, recall)."
            },
            "reasoning": {
                "type": "string", 
                "description": "Brief explanation of why this metric direction was chosen based on the competition description."
            }
        },
        "required": ["is_lower_better", "reasoning"]
    },
    description="Determine whether the evaluation metric should be minimized or maximized based on the competition description."
)