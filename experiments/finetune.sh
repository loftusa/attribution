uv run src/attribution/tune/apps_train.py \
        --model_ckpt BigCode/gpt_345_python_any_license \
        --num_epochs 10 \
        --batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 5e-5 \
        --eval_freq 250 \
        --fp16