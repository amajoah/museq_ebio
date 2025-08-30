# museq_ebio
# µSeq-eBIO — eDNA + CRISPR 바이오센서 위험도 파이프라인 (로컬 실행형)

**목적**: 현장 eDNA 시퀀싱(FASTQ) 및 CRISPR/셀프리 바이오센서 형광 시계열을 간단 처리하여
- 타깃 유전자(독소/유해종) **검출 지표(RPM)**
- 바이오센서 **정량 지표(TTP/기울기 기반 농도 추정)**
- 두 증거의 **확률 결합(베이지안/로지스틱 보정)**
- **위험도/등급** 및 **시각화/JSON** 산출

> 실데이터가 없는 경우 `--simulate`로 합성 FASTQ/시그널을 자동 생성합니다.

## 빠른 시작
```bash
pip install -r requirements.txt
python main.py --simulate
# 또는, 실데이터:
# python main.py --fastq path/to/reads.fastq --biosensor path/to/signal.csv --targets targets.yaml
