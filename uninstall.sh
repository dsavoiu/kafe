#!/bin/bash

# Uninstall script for kafe using pip

function check_pip_python_2_3 () {
    echo "Checking Python version for pip executable '$1'..."
    if ! [[ `which $1` ]]; then
        echo "Pip executable '$1' not found in PATH!"
        return 1
    fi
    local PIP_PYTHON_VERSION=`$1 --version | sed -e "s:.*(python \([2-3]\).*):\1:g"`
    if [[ PIP_PYTHON_VERSION -eq 2 ]]; then
        echo "Pip executable '$1' uses Python 2: ok"
        return 0
    elif [[ PIP_PYTHON_VERSION -eq 3 ]]; then
        echo "Pip executable '$1' uses Python 3: ok"
        return 0
    else
        echo "Pip executable '$1' not found in PATH!"
        return 1
    fi
}

## -- main -- ##

# pip executables to try out
PIP_EXEC_LIST="pip pip2 pip2.7 pip3 pip3.1 pip3.2 pip3.3 pip3.4 pip3.5 pip3.6 pip3.7"

for pip_exec in $PIP_EXEC_LIST; do
    check_pip_python_2_3 $pip_exec
    found_pip_python_2_3=$?
    if [[ found_pip_python_2_3 -eq 0 ]]; then
        break
    fi
done
if [[ found_pip_python_2_3 -ne 0 ]]; then
    echo "None of the executables '$PIP_EXEC_LIST' seem to be valid. Aborting."
    exit 1
fi


echo "Using pip executable '$pip_exec'..."

# Uninstall using pip
$pip_exec uninstall kafe
