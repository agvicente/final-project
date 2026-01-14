#!/usr/bin/env python3
"""
Script para atualizar summaries existentes adicionando balanced_accuracy
Útil para não precisar re-executar todos os experimentos
"""

import json
import pandas as pd
from pathlib import Path
import sys

def update_summary_with_balanced_accuracy(results_dir):
    """
    Atualiza um summary.json adicionando balanced_accuracy baseado no results.json
    
    Args:
        results_dir (Path): Diretório contendo results.json e summary.json
        
    Returns:
        bool: True se atualizado com sucesso, False caso contrário
    """
    results_dir = Path(results_dir)
    
    results_file = results_dir / 'results.json'
    summary_file = results_dir / 'summary.json'
    
    # Verificar se os arquivos existem
    if not results_file.exists():
        print(f"   ⚠️  results.json não encontrado em {results_dir}")
        return False
    
    if not summary_file.exists():
        print(f"   ⚠️  summary.json não encontrado em {results_dir}")
        return False
    
    try:
        # Carregar results.json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not results:
            print(f"   ⚠️  results.json vazio em {results_dir}")
            return False
        
        # Verificar se balanced_accuracy existe nos resultados
        if 'balanced_accuracy' not in results[0]:
            print(f"   ⚠️  balanced_accuracy não encontrado nos resultados de {results_dir}")
            return False
        
        # Calcular balanced_accuracy do summary
        df_results = pd.DataFrame(results)
        best_balanced_accuracy = float(df_results['balanced_accuracy'].max())
        mean_balanced_accuracy = float(df_results['balanced_accuracy'].mean())
        
        # Carregar summary existente
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Verificar se já tem balanced_accuracy (para evitar reprocessamento)
        if 'best_balanced_accuracy' in summary:
            print(f"   ℹ️  balanced_accuracy já existe em {results_dir.name}")
            return True
        
        # Adicionar balanced_accuracy ao summary
        summary['best_balanced_accuracy'] = best_balanced_accuracy
        summary['mean_balanced_accuracy'] = mean_balanced_accuracy
        
        # Salvar summary atualizado
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Atualizado: {results_dir.name}")
        print(f"      Best Balanced Accuracy: {best_balanced_accuracy:.4f}")
        print(f"      Mean Balanced Accuracy: {mean_balanced_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro ao processar {results_dir}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def update_all_summaries(mode='full'):
    """
    Atualiza todos os summaries em experiments/results/test ou experiments/results/full
    
    Args:
        mode (str): 'test' ou 'full'
    """
    results_base = Path('experiments/results') / mode
    
    if not results_base.exists():
        print(f"❌ Diretório não encontrado: {results_base}")
        return 0
    
    print(f"🔍 Buscando summaries em: {results_base}")
    print(f"📋 Modo: {mode.upper()}\n")
    
    # Buscar todas as pastas com timestamp
    algorithm_dirs = []
    for algo_dir in results_base.iterdir():
        if algo_dir.is_dir() and not algo_dir.name.endswith('_consolidation'):
            summary_file = algo_dir / 'summary.json'
            if summary_file.exists():
                algorithm_dirs.append(algo_dir)
    
    if not algorithm_dirs:
        print(f"❌ Nenhum summary encontrado em {results_base}")
        return 0
    
    print(f"📊 Encontrados {len(algorithm_dirs)} summaries para atualizar\n")
    
    # Atualizar cada summary
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, algo_dir in enumerate(algorithm_dirs, 1):
        print(f"[{i}/{len(algorithm_dirs)}] Processando {algo_dir.name}...")
        
        result = update_summary_with_balanced_accuracy(algo_dir)
        
        if result:
            # Verificar se foi atualizado ou já existia
            with open(algo_dir / 'summary.json', 'r') as f:
                summary = json.load(f)
                if 'best_balanced_accuracy' in summary:
                    updated_count += 1
        else:
            error_count += 1
        
        print()
    
    # Resumo final
    print("=" * 60)
    print(f"🎉 ATUALIZAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"✅ Summaries atualizados: {updated_count}")
    print(f"❌ Erros: {error_count}")
    print(f"📊 Total processado: {len(algorithm_dirs)}")
    
    if updated_count > 0:
        print(f"\n💡 Próximo passo: Execute consolidate_results.py para regenerar os gráficos!")
        print(f"   python3 experiments/consolidate_results.py")
    
    return updated_count

def main():
    """Função principal"""
    print("🔧 Script de Atualização de Summaries - Balanced Accuracy")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode not in ['test', 'full']:
            print("❌ Modo deve ser 'test' ou 'full'")
            print("Uso: python3 update_summaries_balanced_accuracy.py [test|full]")
            sys.exit(1)
    else:
        # Auto-detectar qual modo tem mais resultados
        test_base = Path('experiments/results/test')
        full_base = Path('experiments/results/full')
        
        test_count = len([d for d in test_base.iterdir() if d.is_dir()]) if test_base.exists() else 0
        full_count = len([d for d in full_base.iterdir() if d.is_dir()]) if full_base.exists() else 0
        
        if test_count == 0 and full_count == 0:
            print("❌ Nenhum resultado encontrado em experiments/results/")
            sys.exit(1)
        
        mode = 'test' if test_count > full_count else 'full'
        print(f"🔍 Auto-detectado: modo {mode.upper()} ({test_count if mode == 'test' else full_count} pastas)\n")
    
    updated = update_all_summaries(mode)
    sys.exit(0 if updated > 0 else 1)

if __name__ == "__main__":
    main()

