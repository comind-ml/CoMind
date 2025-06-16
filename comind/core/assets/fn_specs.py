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
                "description": "If the code **ran successfully and produced submission.csv on full test set (i.e. not dummy or partial)**, report the value of the **final** validation metric. Otherwise, leave it null."
            },
            "is_lower_better": {
                "type": "boolean", 
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy)."
            }
        },
        "required": [
            "is_bug",
            "summary",
            "output_abs",
            "metric",
            "is_lower_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
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