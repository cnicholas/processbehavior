# Examples Directory

This directory contains working copies of tutorials and example notebooks for learning and experimentation.

## Structure

- **`tom/`** - Tom's working directory for learning ProcessBehavior
  - Contains a copy of `fillweight_analysis_tutorial.ipynb` that can be modified freely
  - Feel free to create additional notebooks here for experimentation

## Usage

### For Tom (or other learners):

1. Navigate to your directory:
   ```bash
   cd examples/tom
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook fillweight_analysis_tutorial.ipynb
   ```

3. Experiment freely! This is your workspace - modify cells, try different parameters, break things and learn.

4. The original tutorial at the root level (`../../fillweight_analysis_tutorial.ipynb`) remains unchanged as a reference.

### Creating Additional Working Copies

```bash
cd examples/tom
cp fillweight_analysis_tutorial.ipynb my_experiment_1.ipynb
jupyter notebook my_experiment_1.ipynb
```

## Notes

- Files in `examples/` directories are meant to be modified and experimented with
- Create as many copies as needed for different experiments
- The original tutorials in the project root serve as clean references
- Consider adding `examples/*/` to `.gitignore` if you don't want to commit your experiments
