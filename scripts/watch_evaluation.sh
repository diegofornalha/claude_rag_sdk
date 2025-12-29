#!/bin/bash
# Script para acompanhar avaliação em tempo real
# Uso: ./scripts/watch_evaluation.sh

LOG_FILE="/tmp/server.log"
REFRESH=2  # segundos entre atualizações

echo "📊 Monitorando Avaliação Atlantyx..."
echo "   Log: $LOG_FILE"
echo "   Ctrl+C para sair"
echo ""

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo "  📊 AVALIAÇÃO ATLANTYX - $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # Buscar última avaliação
    LAST_START=$(grep "Starting evaluation" "$LOG_FILE" 2>/dev/null | tail -1)

    if [ -z "$LAST_START" ]; then
        echo "  ⏳ Aguardando início da avaliação..."
    else
        TIMESTAMP=$(echo "$LAST_START" | grep -oE '"timestamp": "[^"]+"' | cut -d'"' -f4 | cut -d'T' -f2 | cut -d'.' -f1)
        echo "  🚀 Iniciada às: $TIMESTAMP"
        echo ""
        echo "  ┌─────┬────────┬─────────┬──────────┐"
        echo "  │  Q  │ Status │  Score  │ Latência │"
        echo "  ├─────┼────────┼─────────┼──────────┤"

        for i in {1..10}; do
            RESULT=$(grep "Q$i:" "$LOG_FILE" 2>/dev/null | tail -1)

            if echo "$RESULT" | grep -q "✅"; then
                SCORE=$(echo "$RESULT" | grep -oE 'score=[0-9.]+%' | cut -d'=' -f2)
                LATENCY=$(echo "$RESULT" | grep -oE 'latency=[0-9]+ms' | cut -d'=' -f2)
                LATENCY_S=$(echo "scale=1; ${LATENCY%ms}/1000" | bc 2>/dev/null || echo "${LATENCY%ms}ms")
                printf "  │ Q%-2d │   ✅   │ %6s  │  %6ss │\n" "$i" "$SCORE" "$LATENCY_S"
            elif echo "$RESULT" | grep -q "❌"; then
                SCORE=$(echo "$RESULT" | grep -oE 'score=[0-9.]+%' | cut -d'=' -f2)
                printf "  │ Q%-2d │   ❌   │ %6s  │    -     │\n" "$i" "$SCORE"
            elif grep -q "Evaluating Q$i:" "$LOG_FILE" 2>/dev/null; then
                printf "  │ Q%-2d │   🔄   │   ...   │    ...   │\n" "$i"
            else
                printf "  │ Q%-2d │   ⏳   │    -    │    -     │\n" "$i"
            fi
        done

        echo "  └─────┴────────┴─────────┴──────────┘"
        echo ""

        # Verificar se completou
        COMPLETE=$(grep "Evaluation complete" "$LOG_FILE" 2>/dev/null | tail -1)
        if [ -n "$COMPLETE" ]; then
            PASS_RATE=$(echo "$COMPLETE" | grep -oE 'Pass rate: [0-9.]+%' | cut -d' ' -f3)
            echo "  ════════════════════════════════════════"
            echo "  🎉 RESULTADO FINAL: $PASS_RATE"
            echo "  ════════════════════════════════════════"

            # Mostrar recomendações se houver
            echo ""
            echo "  Pressione Ctrl+C para sair ou aguarde nova avaliação..."
        fi
    fi

    sleep $REFRESH
done
