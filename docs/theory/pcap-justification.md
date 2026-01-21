# Justificativa PCAP - Por que Processar PCAPs

**Criado:** 2025-12-17
**Última Atualização:** 2026-01-20
**Capítulo da Dissertação:** 4 - Metodologia

> **Propósito:** Documento de JUSTIFICATIVA METODOLÓGICA para a dissertação. Explica por que os arquivos PCAP do CICIoT2023 devem ser processados diretamente (em vez dos CSVs) para manter a ordem temporal e permitir simulação realista de streaming.

---

## 1. Contexto e Justificativa

### 1.1 Por que Processar PCAPs?

O paper do CICIoT2023 (Neto et al., 2023) revela que os arquivos CSV disponíveis:

> "are combined and **shuffled** into a single dataset (i.e., blended dataset) using PySpark"
> — Seção 5, linha 1839 do paper

**Implicações:**
- ❌ **Ordem temporal perdida** nos CSVs
- ❌ **Concept drift natural impossível** de simular
- ❌ **Streaming realista inviável** sem informação temporal

**Para um trabalho válido com:**
- Streaming real-time via Kafka
- Concept drift natural (não artificial)
- Simulação realista de rede IoT

**É MANDATÓRIO processar os arquivos PCAP originais.**

### 1.2 Estrutura do Dataset CICIoT2023

| Diretório | Conteúdo | Tamanho |
|-----------|----------|---------|
| **PCAP/** | Tráfego original em .pcap | ~548 GB |
| **CSV/** | Features extraídas (shuffled) | ~13 GB |
| **Example/** | Jupyter notebook de exemplo | - |
| **Supplementary/** | Scripts de coleta | - |

**Fonte:** [UNB CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---

## 2. Arquivos CSV vs PCAP: Comparação Detalhada

### 2.1 Arquivos CSV (Atuais)

**Estrutura:**
- 169 arquivos CSV
- 47 features extraídas
- Janelamento: 10 ou 100 pacotes (dependendo do ataque)
- Timestamp removido
- Dados embaralhados (shuffled)

**O que foi feito no processamento original:**
```
PCAP → TCPDUMP (split 10MB) → DPKT (feature extraction) →
→ Pandas (windowing) → PySpark (shuffle + merge) → CSV
```

**Limitações para nosso projeto:**
- Sem informação temporal
- Concept drift só pode ser simulado artificialmente
- Não sabemos quando cada ataque ocorreu

### 2.2 Arquivos PCAP (Necessários)

**Estrutura:**
- ~548 GB de dados brutos
- Divididos em chunks de 10 MB (via TCPDUMP)
- Preservam timestamps reais
- Um arquivo por cenário de ataque + benign

**Ferramentas usadas pelos autores:**
- [Wireshark](https://www.wireshark.org/) - Captura original
- [Mergecap](https://www.wireshark.org/docs/man-pages/mergecap.html) - Unificação de capturas
- [TCPDUMP](https://www.tcpdump.org/) - Split em chunks
- [DPKT](https://dpkt.readthedocs.io/) - Extração de features

---

## 3. Ferramentas para Processamento PCAP

### 3.1 Leitura e Parsing de PCAP

| Ferramenta | Linguagem | Performance | Features | Recomendação |
|------------|-----------|-------------|----------|--------------|
| **NFStream** | Python | Alta | 48+ features, DPI | ⭐ Recomendado |
| **DPKT** | Python | Alta | Baixo nível | Usado no paper |
| **Scapy** | Python | Média | Flexível | Prototipagem |
| **PyShark** | Python | Baixa | Completo | Análise detalhada |
| **cicflowmeter** | Python | Média | CIC-compatible | Alternativa |

#### NFStream (Recomendado)
```python
from nfstream import NFStreamer

# Processar PCAP com extração de features
streamer = NFStreamer(source="capture.pcap",
                      statistical_analysis=True,
                      splt_analysis=10,  # Early flow features
                      n_dissections=20)

for flow in streamer:
    # flow contém 48+ features estatísticas
    print(flow.to_namedtuple())
```

**Vantagens:**
- Performance otimizada (C backend)
- 48 features estatísticas pré-definidas
- Deep Packet Inspection (nDPI)
- Suporte a extensões customizadas
- Exporta direto para Pandas/CSV

**Referências:**
- [NFStream Documentation](https://www.nfstream.org/)
- [NFStream GitHub](https://github.com/nfstream/nfstream)
- [NFStream PyPI](https://pypi.org/project/nfstream/)

#### DPKT (Usado no paper original)
```python
import dpkt

with open('capture.pcap', 'rb') as f:
    pcap = dpkt.pcap.Reader(f)
    for timestamp, buf in pcap:
        eth = dpkt.ethernet.Ethernet(buf)
        # Processar pacote...
```

**Referências:**
- [DPKT Documentation](https://dpkt.readthedocs.io/)
- [DPKT GitHub](https://github.com/kbandla/dpkt)

#### Scapy (Prototipagem)
```python
from scapy.all import PcapReader

# Leitura eficiente (não carrega tudo na memória)
with PcapReader('capture.pcap') as pcap:
    for pkt in pcap:
        # Processar pacote...
        pass
```

**Referências:**
- [Scapy Documentation](https://scapy.readthedocs.io/)
- [Network Traffic Analysis with Scapy](https://cylab.be/blog/245/network-traffic-analysis-with-python-scapy-and-some-machine-learning)

### 3.2 Replay de PCAP (Simulação de Rede)

#### tcpreplay
```bash
# Replay na velocidade original
tcpreplay -i eth0 capture.pcap

# Replay 2x mais rápido
tcpreplay -x 2.0 -i eth0 capture.pcap

# Replay a 100 Mbps
tcpreplay -M 100 -i eth0 capture.pcap

# Replay controlado (um pacote por vez)
tcpreplay -o -i eth0 capture.pcap
```

**Opções de controle de velocidade:**
| Flag | Descrição |
|------|-----------|
| `-x N` | Multiplicador de velocidade (2.0 = 2x) |
| `-p N` | Pacotes por segundo |
| `-M N` | Mbps fixo |
| `-t` | Máxima velocidade possível |
| `-o` | Um pacote por vez (manual) |

**Referências:**
- [tcpreplay Official](https://tcpreplay.appneta.com/)
- [tcpreplay Man Page](https://tcpreplay.appneta.com/wiki/tcpreplay-man.html)
- [tcpreplay GitHub](https://github.com/appneta/tcpreplay)

### 3.3 Alternativas ao CICFlowMeter

O CICFlowMeter original (Java) tem problemas conhecidos. Alternativas Python:

| Ferramenta | Status | Compatibilidade CIC |
|------------|--------|---------------------|
| **NFStream** | Ativo, 2024 | Parcial (diferentes features) |
| **cicflowmeter (Python)** | Ativo | Alta |
| **LycoSTand** | 2024 | Alta (corrige bugs do CIC) |
| **NTLFlowLyzer** | 2024 | Alta (corrige bugs do CIC) |
| **HERA** | 2025 | Customizável |

**Referências:**
- [cicflowmeter Python](https://github.com/hieulw/cicflowmeter)
- [HERA Tool (2025)](https://arxiv.org/html/2501.07475v1)
- [Comparison of Feature Extraction Tools (2025)](https://arxiv.org/pdf/2501.13004)

---

## 4. Estratégias de Processamento

### 4.1 Abordagem A: Processamento Batch (Simples)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PCAPs     │────▶│  NFStream/  │────▶│   CSV com   │
│  (548 GB)   │     │    DPKT     │     │  Timestamp  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Ordenar   │
                    │   por time  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Stream    │
                    │ Simulado    │
                    └─────────────┘
```

**Vantagens:**
- Simples de implementar
- Pode processar em paralelo
- Resultado reutilizável

**Desvantagens:**
- Requer ~500GB+ de disco
- Processamento inicial longo
- Não é "real-time" verdadeiro

### 4.2 Abordagem B: Pipeline Real-time

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PCAPs     │────▶│  tcpreplay  │────▶│  Interface  │
│  (subset)   │     │  (speedup)  │     │   Virtual   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────┐
                    │           KAFKA                   │
                    │  ┌─────────┐    ┌─────────────┐  │
                    │  │ Sniffer │───▶│ raw-packets │  │
                    │  └─────────┘    └─────────────┘  │
                    └──────────────────────────────────┘
                                               │
                                               ▼
                    ┌─────────────────────────────────┐
                    │     Consumer (Feature Extract)   │
                    │  - Janelamento (10/100 pkts)    │
                    │  - Extração de features         │
                    │  - MicroTEDAclus                │
                    └─────────────────────────────────┘
```

**Vantagens:**
- Real-time verdadeiro
- Simula produção real
- Métricas de latência válidas

**Desvantagens:**
- Complexidade maior
- Requer interface de rede virtual
- Debugging mais difícil

### 4.3 Abordagem C: Híbrida (Recomendada para MVP)

```
FASE 1: Pré-processamento
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PCAPs     │────▶│  NFStream   │────▶│   Parquet   │
│  (subset)   │     │             │     │ c/ Timestamp│
└─────────────┘     └─────────────┘     └─────────────┘

FASE 2: Streaming Simulado
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Parquet    │────▶│   Producer  │────▶│    KAFKA    │
│  ordenado   │     │  (tempo     │     │             │
│             │     │   real)     │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Vantagens:**
- Combina simplicidade com realismo
- Parquet é mais eficiente que CSV
- Pode acelerar/desacelerar o tempo
- Debugging mais fácil

---

## 5. Features a Extrair

### 5.1 Features do Paper Original (47)

As 47 features extraídas no CICIoT2023:

| # | Feature | Descrição |
|---|---------|-----------|
| 1 | flow_duration | Duração do fluxo |
| 2 | Header_Length | Comprimento do cabeçalho |
| 3 | Protocol_Type | IP, UDP, TCP, IGMP, ICMP |
| 4 | Duration | Time-to-Live (TTL) |
| 5 | Rate | Taxa de transmissão |
| 6 | Srate | Taxa de pacotes de saída |
| 7 | Drate | Taxa de pacotes de entrada |
| 8-15 | *_flag_number | Flags TCP (fin, syn, rst, psh, ack, ece, cwr) |
| 16-20 | *_count | Contadores de flags |
| 21-34 | Protocol flags | HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IPv, LLC |
| 35-41 | Statistics | Tot_sum, Min, Max, AVG, Std, Tot_size, IAT |
| 42 | Number | Número de pacotes no fluxo |
| 43 | Magnitude | (avg_in + avg_out)^0.5 |
| 44 | Radius | (var_in + var_out)^0.5 |
| 45 | Covariance | Covariância in/out |
| 46 | Variance | var_in / var_out |
| 47 | Weight | n_in × n_out |

### 5.2 Features Adicionais para Concept Drift

Para detectar concept drift, considerar adicionar:

| Feature | Propósito |
|---------|-----------|
| **timestamp** | Ordem temporal (NÃO remover!) |
| **window_id** | Identificador da janela |
| **src_ip_entropy** | Diversidade de IPs origem |
| **dst_port_entropy** | Diversidade de portas destino |
| **packet_rate_delta** | Mudança na taxa de pacotes |

---

## 6. Plano de Implementação

### 6.1 Fase Imediata (Semana Atual)

1. **Verificar acesso aos PCAPs**
   - Acessar [UNB CIC Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
   - Verificar se PCAPs estão disponíveis para download
   - Se não, solicitar acesso aos autores

2. **Escolher subset inicial**
   - ~50 GB representativo
   - Incluir: Benign, DDoS (maior), Mirai (médio), Recon (menor)

3. **Setup de ferramentas**
   ```bash
   pip install nfstream dpkt scapy pandas pyarrow
   ```

### 6.2 Fase de Desenvolvimento (Semanas 2-3)

1. **Implementar extrator de features**
   - Baseado em NFStream
   - Preservar timestamp
   - Salvar em Parquet

2. **Validar contra CSV oficial**
   - Comparar features extraídas
   - Documentar diferenças

3. **Criar producer Kafka**
   - Ler Parquet ordenado
   - Simular tempo real

### 6.3 Fase de Integração (Semanas 4+)

1. **Integrar com MicroTEDAclus**
2. **Implementar métricas de concept drift**
3. **Benchmark de performance**

---

## 7. Considerações de Performance

### 7.1 Processamento de 548 GB

| Abordagem | Tempo Estimado | Recursos |
|-----------|----------------|----------|
| Sequencial | ~24-48 horas | 1 CPU, 16GB RAM |
| Paralelo (8 cores) | ~4-8 horas | 8 CPU, 32GB RAM |
| Distribuído (Spark) | ~1-2 horas | Cluster |

**Recomendação:** Processar subset de ~50GB primeiro para validação.

### 7.2 Armazenamento

| Formato | Compressão | Tamanho (50GB PCAP → ) |
|---------|------------|------------------------|
| CSV | Nenhuma | ~15 GB |
| CSV.gz | gzip | ~3 GB |
| Parquet | Snappy | ~2 GB |
| Parquet | zstd | ~1.5 GB |

**Recomendação:** Parquet com compressão Snappy (bom balanço velocidade/tamanho).

---

## 8. Referências

### Papers

1. **CICIoT2023 Dataset Paper:**
   - Neto, E.C.P., et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment." *Sensors*, 23(13), 5941.
   - [MDPI Link](https://www.mdpi.com/1424-8220/23/13/5941)
   - [PDF](https://www.mdpi.com/1424-8220/23/13/5941/pdf)

2. **Comparison of Feature Extraction Tools (2025):**
   - [arXiv](https://arxiv.org/pdf/2501.13004)

3. **HERA Tool (2025):**
   - [arXiv](https://arxiv.org/html/2501.07475v1)

### Ferramentas

| Ferramenta | Documentação | GitHub |
|------------|--------------|--------|
| NFStream | [nfstream.org](https://www.nfstream.org/) | [nfstream/nfstream](https://github.com/nfstream/nfstream) |
| DPKT | [dpkt.readthedocs.io](https://dpkt.readthedocs.io/) | [kbandla/dpkt](https://github.com/kbandla/dpkt) |
| Scapy | [scapy.readthedocs.io](https://scapy.readthedocs.io/) | [secdev/scapy](https://github.com/secdev/scapy) |
| tcpreplay | [tcpreplay.appneta.com](https://tcpreplay.appneta.com/) | [appneta/tcpreplay](https://github.com/appneta/tcpreplay) |
| cicflowmeter (Python) | [PyPI](https://pypi.org/project/cicflowmeter/) | [hieulw/cicflowmeter](https://github.com/hieulw/cicflowmeter) |
| PyPCAPKit | [pypcapkit.jarryshaw.me](https://pypcapkit.jarryshaw.me/) | [JarryShaw/PyPCAPKit](https://github.com/JarryShaw/PyPCAPKit) |

### Dataset

- **CICIoT2023 Official:** [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- **IEEE DataPort:** [CICIoT2023 Dataset](https://ieee-dataport.org/documents/ciciot2023-dataset)

### Tutoriais

- [Network Traffic Analysis with Python & Scapy](https://cylab.be/blog/245/network-traffic-analysis-with-python-scapy-and-some-machine-learning)
- [Analyzing Packet Captures with Python](https://vnetman.github.io/pcap/python/pyshark/scapy/libpcap/2018/10/25/analyzing-packet-captures-with-python-part-1.html)
- [Streaming Network Packets to Kafka](https://medium.com/codex/streaming-live-network-packets-data-in-to-spark-streaming-using-kafka-25c8b5f58181)
- [tcpreplay Tutorial](https://www.techtarget.com/searchsecurity/tutorial/How-to-use-tcpreplay-to-replay-network-packet-files)

---

## 9. Decisões e Status

| Decisão | Opções | Status |
|---------|--------|--------|
| Acesso aos PCAPs | Download direto vs Solicitar autores | ✅ **Disponível via SSH** |
| Ferramenta principal | NFStream vs DPKT vs cicflowmeter | NFStream (tentativa) |
| Tamanho do subset inicial | 10GB, 50GB, 100GB | 50GB (sugerido) |
| Formato de armazenamento | CSV vs Parquet | Parquet (recomendado) |
| Abordagem de streaming | Batch → Kafka vs Real-time | Híbrida (recomendado) |

**Nota:** Todos os arquivos PCAP (~548GB) estão disponíveis em máquina remota acessível via SSH.

---

**Este documento serve como referência para a implementação do pipeline de processamento PCAP.**

*Última atualização: 2025-12-17*
