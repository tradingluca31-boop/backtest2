# Vérification des Formules Mathématiques - Backtest Analyzer

## 1. SHARPE RATIO

### Formule actuelle (ligne 364)
```python
metrics['Sharpe'] = annual_return / vol if vol > 0 else 0
```

### Formule correcte selon sources officielles
**Source:** https://www.investopedia.com/terms/s/sharperatio.asp
```
Sharpe = (Rp - Rf) / σp
```
Où:
- Rp = Rendement du portefeuille
- Rf = Taux sans risque (généralement 0 pour simplifier)
- σp = Écart-type des rendements

### ⚠️ PROBLÈME IDENTIFIÉ
La volatilité (`vol`) utilisée est l'écart-type **des returns individuels**, mais elle n'est **PAS annualisée**.

### ✅ CORRECTION NÉCESSAIRE
```python
# Annualiser la volatilité en fonction de la fréquence
annual_vol = vol * np.sqrt(trades_per_year)
metrics['Sharpe'] = annual_return / annual_vol if annual_vol > 0 else 0
```

---

## 2. SORTINO RATIO

### Formule actuelle (ligne 367-369)
```python
negative_returns = returns[returns < 0]
downside_std = negative_returns.std() if len(negative_returns) > 0 else vol
metrics['Sortino'] = annual_return / downside_std if downside_std > 0 else 0
```

### Formule correcte
**Source:** https://www.investopedia.com/terms/s/sortinoratio.asp
```
Sortino = (Rp - Rf) / σd
```
Où σd = Downside deviation (écart-type des rendements négatifs uniquement)

### ⚠️ PROBLÈME IDENTIFIÉ
Même problème: `downside_std` n'est PAS annualisé

### ✅ CORRECTION NÉCESSAIRE
```python
annual_downside_std = downside_std * np.sqrt(trades_per_year)
metrics['Sortino'] = annual_return / annual_downside_std if annual_downside_std > 0 else 0
```

---

## 3. CALMAR RATIO

### Formule actuelle (ligne 378)
```python
metrics['Calmar'] = metrics['CAGR'] / metrics['Max_Drawdown'] if metrics['Max_Drawdown'] > 0 else 0
```

### Formule correcte
**Source:** https://www.investopedia.com/terms/c/calmarratio.asp
```
Calmar = CAGR / |Max Drawdown|
```

### ✅ CORRECT
La formule est bonne, le Max Drawdown est déjà en valeur absolue.

---

## 4. MAX DRAWDOWN

### Formule actuelle (ligne 372-375)
```python
cumulative_returns = (1 + returns).cumprod()
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max
metrics['Max_Drawdown'] = abs(drawdown.min())
```

### Formule correcte
**Source:** https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp
```
MDD = (Trough Value - Peak Value) / Peak Value
```

### ✅ CORRECT
La formule est correcte.

---

## 5. CAGR (Compound Annual Growth Rate)

### Formule actuelle (ligne 341-347)
```python
total_return = (1 + returns).prod() - 1
time_period = (returns.index[-1] - returns.index[0]).days / 365.25
if time_period > 0 and total_return > -1:
    metrics['CAGR'] = (1 + total_return) ** (1/time_period) - 1
else:
    metrics['CAGR'] = total_return
```

### Formule correcte
**Source:** https://www.investopedia.com/terms/c/cagr.asp
```
CAGR = (Ending Value / Beginning Value)^(1/years) - 1
```

### ⚠️ PROBLÈME POTENTIEL
Si les données sont des **trades** (pas des returns journaliers), il faut utiliser l'equity curve réelle, pas le produit cumulé des returns.

### ✅ CORRECTION POUR TRADES MT5
```python
if hasattr(self, 'original_trades_data') and 'time_close' in self.original_trades_data.columns:
    # Pour trades MT5, utiliser l'equity curve réelle
    initial_equity = 10000
    final_equity = initial_equity + self.original_trades_data['profit'].sum()
    total_return = (final_equity - initial_equity) / initial_equity
    time_period = (returns.index[-1] - returns.index[0]).days / 365.25
    metrics['CAGR'] = (1 + total_return) ** (1/time_period) - 1 if time_period > 0 else total_return
else:
    # Pour returns, utiliser la méthode actuelle
    total_return = (1 + returns).prod() - 1
    ...
```

---

## 6. PROBABILISTIC SHARPE RATIO

### Formule actuelle (ligne 448-473)
Utilise une approximation basée sur des seuils arbitraires.

### Formule correcte
**Source:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
```
PSR = Φ(√(n-1) * SR / √(1 - γ3·SR + (γ4-1)/4·SR²))
```
Où:
- Φ = Fonction de distribution cumulative normale
- n = Nombre d'observations
- SR = Sharpe Ratio
- γ3 = Skewness
- γ4 = Kurtosis

### ⚠️ PROBLÈME MAJEUR
La formule actuelle est une approximation très simplifiée qui ne suit pas la littérature académique.

### ✅ CORRECTION NÉCESSAIRE
```python
from scipy import stats as scipy_stats

n = len(returns)
sr = metrics['Sharpe']
skew = metrics['Skewness']
kurt = metrics['Kurtosis']

if n > 1 and metrics['Volatility'] > 0:
    # Formule de Haircut Sharpe Ratio (Bailey & López de Prado)
    denominator = np.sqrt(1 - skew*sr + ((kurt-1)/4)*sr**2)
    psr_stat = np.sqrt(n-1) * sr / denominator if denominator > 0 else 0
    psr = scipy_stats.norm.cdf(psr_stat)
    metrics['Probabilistic_Sharpe_Ratio'] = psr
else:
    metrics['Probabilistic_Sharpe_Ratio'] = 0.5
```

---

## 7. PREMIER TRADE (18 JANVIER)

### Problème identifié
Le code trie les données par `time_close` mais ne vérifie pas si la **première date affichée** correspond bien au premier trade chronologique.

### Vérification nécessaire
Ajouter un affichage debug:
```python
# Ligne 3219, après conversion
df['time_close_dt'] = pd.to_datetime(df['time_close'], unit='s', errors='coerce')

# AJOUTER DEBUG
first_trade_date = df['time_close_dt'].min()
last_trade_date = df['time_close_dt'].max()
st.info(f"📅 Premier trade: {first_trade_date.strftime('%d/%m/%Y %H:%M')}")
st.info(f"📅 Dernier trade: {last_trade_date.strftime('%d/%m/%Y %H:%M')}")
st.info(f"📊 Nombre total de trades: {len(df)}")
```

---

## RÉSUMÉ DES CORRECTIONS PRIORITAIRES

### 🔴 URGENT
1. **Sharpe Ratio:** Annualiser la volatilité
2. **Sortino Ratio:** Annualiser la downside deviation
3. **Probabilistic Sharpe Ratio:** Utiliser la vraie formule académique
4. **Premier trade:** Ajouter debug pour vérifier la date

### 🟡 IMPORTANT
1. **CAGR:** Vérifier le calcul pour trades MT5 (utiliser equity réelle)
2. **Returns calculation:** S'assurer que les returns sont bien calculés depuis l'equity curve

### 🟢 OPTIONNEL
1. Ajouter des warnings si les données sont insuffisantes (< 30 trades)
2. Implémenter les tests statistiques (White Reality Check, Hansen SPA)

---

## SOURCES DE RÉFÉRENCE

1. **Sharpe/Sortino/Calmar:** https://www.investopedia.com/
2. **Probabilistic Sharpe Ratio:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
3. **Deflated Sharpe Ratio:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
4. **QuantStats (pour comparaison):** https://github.com/ranaroussi/quantstats
5. **ISLR:** https://www.statlearning.com/
6. **MQL5 Python:** https://www.mql5.com/en/docs/python_metatrader5
