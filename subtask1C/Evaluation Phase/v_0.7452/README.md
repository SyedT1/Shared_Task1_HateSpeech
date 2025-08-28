## Implementation
  1. Initialization: Random uniform perturbation in        
  [-ε, ε], normalized to ε-ball
  2. K-step loop: K=2 adversarial steps with gradient      
  accumulation
  3. Gradient accumulation: Gradients accumulated
  across K steps (loss scaled by 1/K)
  4. Perturbation update: Gradient ascent with
  normalization and projection back to ε-ball
  5. Hook-based implementation: Works with your custom     
  MultiTaskModel

  🔧 Memory Optimizations:

  - Batch size: 4 (with gradient accumulation = 4 for      
  effective batch of 16)
  - K reduced from 3 to 2
  - FP16 disabled (incompatible with current FreeLB        
  implementation)
  - Fallback mechanism to standard training if OOM

  📊 Hyperparameters:

  - K = 2: Number of adversarial steps
  - ε = 0.2: Maximum perturbation norm
  - α = 0.02: Step size for gradient ascent
  - Learning rate = 2e-5
  - Epochs = 2

  The implementation correctly follows the FreeLB paper    
   while being optimized for your memory constraints.      
  Try running it now - it should work without the FP16     
  scaler issue!
