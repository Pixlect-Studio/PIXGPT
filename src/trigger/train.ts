import { task } from "@trigger.dev/sdk";
import { python } from "@trigger.dev/python";

export const trainPixGptTask = task({
  id: "train-pixgpt-model",
  
  // Trigger.dev limits how long a task can run. 
  // Set this to the maximum allowed by your plan (e.g., 600 seconds = 10 minutes).
  maxDuration: 600, 
  
  run: async (payload: { 
    source_type: string, 
    source_val: string, 
    offset?: number, 
    batch_size?: number 
  }) => {
    
    console.log(`Starting training for ${payload.source_val}...`);

    // Execute the Python script and pass the JSON payload as a string argument
    const result = await python.runScript(
      "./python/auto_train.py", 
      [JSON.stringify(payload)]
    );

    console.log("Python Script Output:");
    console.log(result.stdout);

    if (result.stderr) {
      console.error("Python Script Errors:");
      console.error(result.stderr);
    }

    return {
      stdout: result.stdout,
      stderr: result.stderr,
      success: true
    };
  },
});