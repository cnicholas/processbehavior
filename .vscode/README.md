# VS Code Configuration for processbehavior Project

## 📁 Configuration Files

This directory contains VS Code configuration files for the processbehavior project.

### Core Configuration
- **`settings.json`** - Main VS Code settings including Python test configuration
- **`launch.json`** - Debug configurations for running and debugging tests
- **`tasks.json`** - Custom tasks for running specific test suites

## 🚀 Quick Start

### 1. Open the Testing Panel
Click the beaker/flask icon in the left sidebar, or press `Cmd+Shift+P` and search for "Test: Focus on Test Explorer View"

### 2. Discover Tests
Click the refresh icon in the Testing panel. VS Code will automatically find all test files in the `tests/` directory.

### 3. Run Tests
- **All tests**: Click the play button at the top
- **Specific test**: Click the play button next to any test

### 4. Debug Tests
Click the bug icon next to any test to run it in debug mode with breakpoints.

## 📋 Available Tasks

Run tasks via `Cmd+Shift+P` → "Tasks: Run Task":
- Run All Tests
- Run SDS1 Test
- Run SDS2 Test
- Run Analysis Dataset Tests
- Run Analysis Specifications Tests

## 🐛 Debug Configurations

Available in the Run and Debug panel (`Cmd+Shift+D`):
- **Python: Current File** - Debug the currently open Python file
- **Python: Debug Tests** - Debug all tests
- **Python: Debug Current Test File** - Debug tests in the current file

## 📝 What's Configured

### Test Discovery
- Framework: **unittest**
- Test directory: `tests/`
- Test pattern: `test_*.py`
- Python interpreter: `./venv/bin/python`

### Logging
All tests use Python's logging module with:
- Timestamps for each log entry
- Logger names to identify source
- INFO level for test progress
- DEBUG level for detailed output

## 🔧 Customization

You can modify the configuration files to suit your needs:
- Edit `settings.json` for test discovery settings
- Edit `launch.json` for debugging options
- Edit `tasks.json` to add custom test running tasks

## 📚 Learn More

- Read `TESTING_GUIDE.md` for detailed usage instructions
- Check `KEYBOARD_SHORTCUTS.md` for productivity tips
- See the [VS Code Python Testing docs](https://code.visualstudio.com/docs/python/testing)

## ✅ Verification

To verify everything is working:

1. Open the Testing panel (beaker icon)
2. Click refresh
3. You should see all test files and methods listed
4. Click play on any test to run it

If tests don't appear, try:
- Reload the window: `Cmd+Shift+P` → "Developer: Reload Window"
- Check the Python interpreter is correctly set to `./venv/bin/python`
- Check the Output panel → Python Test Log for errors
