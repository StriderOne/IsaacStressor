#!/bin/bash

# Пути (замените на свои)
# ISAAC_PYTHON="python"  # Python IsaacSim
ISAAC_PYTHON="/workspace/isaaclab/isaaclab.sh -p" 
REGISTER_SCRIPT="scripts/tools/env_registration.py"  # Ваш скрипт регистрации сред
TARGET_SCRIPT="$1"  # Первый аргумент — запускаемый скрипт (например, record_demos.py)

# Проверки
if [ ! -f "$REGISTER_SCRIPT" ]; then
    echo "Error: Registration script not found at $REGISTER_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Error: Target script not found at $TARGET_SCRIPT" >&2
    exit 1
fi

# Создаём временный файл
TMP_SCRIPT=$(mktemp /tmp/isaac_runner.XXXXXX.py)

# Вставляем в него:
# 1. Код регистрации сред
# 2. Код целевого скрипта
cat "$REGISTER_SCRIPT" "$TARGET_SCRIPT" > "$TMP_SCRIPT"

# Запускаем через Python IsaacSim и передаём аргументы (кроме первого)
$ISAAC_PYTHON "$TMP_SCRIPT" "${@:2}"

# Удаляем временный файл
rm -f "$TMP_SCRIPT"