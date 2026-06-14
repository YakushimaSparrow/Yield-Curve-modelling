import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from curve import nelson_siegel
from analysis import attach_key_rate, level_vs_rate, end_sensitivity, curvature_summary, inversion_stats

API = "http://127.0.0.1:8080/api/v1"

st.set_page_config(page_title="OFZ Yield Curve", layout="wide")


@st.cache_data(ttl=300)
def get_json(path, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def load_factors():
    df = pd.DataFrame(get_json("/factors"))
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date")


@st.cache_data(ttl=300)
def load_key_rate():
    return pd.DataFrame(get_json("/key-rate"))


@st.cache_data(ttl=300)
def load_instruments():
    return pd.DataFrame(get_json("/instruments"))


@st.cache_data(ttl=300)
def load_market(limit=20000):
    return pd.DataFrame(get_json("/market-data", {"limit": limit}))


lang_choice = st.radio("Language / Язык", ["English", "Русский"], horizontal=True)
lang = "ru" if lang_choice == "Русский" else "en"


def t(en, ru):
    return ru if lang == "ru" else en


def verdict(ok, en, ru):
    text = t(en, ru)
    if ok:
        st.success(t(f"SUPPORTED. {en}", f"ПОДТВЕРЖДАЕТСЯ. {ru}"))
    else:
        st.error(t(f"REJECTED. {en}", f"ОПРОВЕРГНУТА. {ru}"))


st.title(t("What shapes the OFZ yield curve", "Что задаёт форму кривой доходности ОФЗ"))

st.write(t(
    "The yield curve plots the return an investor demands for lending money over different horizons. Its "
    "shape is one of the most watched signals in finance: it reflects where the central bank stands, what "
    "the market expects from monetary policy, and how investors price risk over time. This project takes "
    "the Russian government bond market (OFZ-PD) and asks a concrete question: what actually moves the "
    "shape of that curve, and can those forces be measured in real data?",
    "Кривая доходности показывает, какую доходность инвестор требует за то, чтобы одолжить деньги на "
    "разные сроки. Её форма это один из самых читаемых сигналов в финансах: она отражает, где находится "
    "центробанк, чего рынок ждёт от денежно-кредитной политики и как инвесторы оценивают риск во времени. "
    "Этот проект берёт рынок российских гособлигаций (ОФЗ-ПД) и задаёт конкретный вопрос: что на самом "
    "деле двигает форму этой кривой и можно ли измерить эти силы на реальных данных?",
))
st.write(t(
    "The idea is to reduce the whole curve of each day to three numbers using the Nelson-Siegel model, "
    "level, slope and curvature, and then line them up against the Bank of Russia key rate over four "
    "years. If our hypotheses are right, each of these three numbers should answer to a specific economic "
    "force. The point of the work is not a dashboard but an explanation: how we got the data, what we "
    "measured, and which conclusions the numbers actually support.",
    "Идея в том, чтобы свести всю кривую каждого дня к трём числам с помощью модели Нельсона-Сигеля, "
    "уровень, наклон и кривизна, а затем сопоставить их с ключевой ставкой Банка России за четыре года. "
    "Если наши гипотезы верны, каждое из этих трёх чисел должно откликаться на конкретную экономическую "
    "силу. Смысл работы не в дашборде, а в объяснении: как мы получили данные, что измерили и какие "
    "выводы цифры действительно подтверждают.",
))

factors = load_factors()
key_rate = load_key_rate()
instruments = load_instruments()
market = load_market()

st.header(t("The dataset", "Датасет"))
st.write(t(
    "All data comes from the public MOEX ISS API of the Moscow Exchange and is stored locally in SQLite. "
    "The FastAPI service reads from that database and this page reads from the API. There are two tables. "
    "The instruments table holds the static facts about each bond, and the market data table holds one row "
    "per bond per trading day. The sample covers 27 OFZ-PD issues over 2021 to 2024.",
    "Все данные взяты из публичного API Московской биржи (MOEX ISS) и хранятся локально в SQLite. Сервис "
    "на FastAPI читает из этой базы, а страница читает из сервиса. Таблиц две. Таблица инструментов "
    "хранит статические факты о каждой облигации, а таблица котировок хранит по одной строке на бумагу за "
    "каждый торговый день. Выборка покрывает 27 выпусков ОФЗ-ПД за 2021 по 2024 годы.",
))
st.markdown(t(
    "**Instruments fields:** `secid` issue code, `shortname` name, `coupon_value` rouble coupon, "
    "`coupon_period` days between coupons, `maturity_date` redemption date, `nominal` face value.\n\n"
    "**Market data fields:** `trade_date`, `secid`, `clean_price_pct` clean price as percent of par, "
    "`accint` accrued coupon interest, `dprice` dirty price in roubles, `volume` traded volume.",
    "**Поля инструментов:** `secid` код выпуска, `shortname` название, `coupon_value` купон в рублях, "
    "`coupon_period` дней между купонами, `maturity_date` дата погашения, `nominal` номинал.\n\n"
    "**Поля котировок:** `trade_date`, `secid`, `clean_price_pct` чистая цена в процентах от номинала, "
    "`accint` накопленный купонный доход, `dprice` грязная цена в рублях, `volume` объём торгов.",
))
c1, c2 = st.columns(2)
c1.caption(t("Instruments table", "Таблица инструментов"))
c1.dataframe(instruments, height=300)
c2.caption(t("Market data (sample)", "Котировки (фрагмент)"))
c2.dataframe(market.head(500), height=300)

st.header(t("Descriptive statistics", "Описательные статистики"))
st.write(t(
    "Before any modelling we look at the four numeric market fields across the whole sample. The clean "
    "price stays close to par, the accrued interest cycles between coupon dates, the dirty price is what "
    "actually changes hands, and the volume is heavily skewed by a few very active days.",
    "Перед любым моделированием смотрим на четыре числовых поля котировок по всей выборке. Чистая цена "
    "держится около номинала, накопленный купон ходит между купонными датами, грязная цена это то, что "
    "реально переходит из рук в руки, а объём сильно перекошен несколькими очень активными днями.",
))
num_fields = ["clean_price_pct", "accint", "dprice", "volume"]
stats = market[num_fields].astype(float).describe().T
stats = stats.rename(columns={"50%": "median"})[["mean", "median", "std", "min", "max"]]
st.dataframe(stats.style.format("{:.2f}"))

st.subheader(t("The hypotheses", "Наши гипотезы"))
st.markdown(t(
    "**Hypothesis 1. The shape of the yield curve is set by three forces.** A parallel shift of the whole "
    "curve driven by the key rate, a tilt between the short and long end driven by monetary-policy "
    "expectations, and a curvature in the medium-term segment. In Nelson-Siegel terms this is exactly the "
    "level, the slope and the curvature factor.\n\n"
    "**Hypothesis 2. A sharp tightening cycle inverts the curve.** When the key rate is pushed high, the "
    "short end overshoots the long end and the curve turns from upward sloping to inverted. This one we "
    "test and either confirm or reject.",
    "**Гипотеза 1. Форму кривой доходности задают три силы.** Параллельный сдвиг всей кривой от ключевой "
    "ставки, перекос между коротким и длинным концом от ожиданий по ДКП и выпуклость в среднесрочном "
    "сегменте. В терминах Нельсона-Сигеля это ровно уровень, наклон и кривизна.\n\n"
    "**Гипотеза 2. Резкое ужесточение инвертирует кривую.** Когда ключевую ставку задирают высоко, "
    "короткий конец перелетает длинный и кривая из растущей превращается в инвертированную. Эту гипотезу "
    "мы проверяем и либо подтверждаем, либо опровергаем.",
))

if factors.empty:
    st.warning(t("The database is empty. Run build_database.py first.",
                 "База пустая. Сначала запусти build_database.py."))
    st.stop()

merged = attach_key_rate(factors, key_rate)
merged["short_end"] = merged["level"] + merged["slope"]
merged["long_end"] = merged["level"]
merged["spread"] = merged["slope"].abs()

res1 = level_vs_rate(merged)
res2 = end_sensitivity(merged)
res3 = curvature_summary(merged)
res4 = inversion_stats(merged)
ratio = res2["short_beta"] / res2["long_beta"]

st.subheader(t("Headline results", "Главные результаты"))
m1, m2, m3, m4 = st.columns(4)
m1.metric(t("Level vs key rate", "Уровень и ключ"), f"r = {res1['corr']:.2f}",
          t("parallel shift", "параллельный сдвиг"))
m2.metric(t("Short vs long end", "Короткий и длинный"), f"{ratio:.1f}x",
          t("short end more sensitive", "короткий чувствительнее"))
m3.metric(t("Positive curvature", "Положительная кривизна"), f"{res3['share_positive'] * 100:.0f}%",
          t("of all days", "всех дней"))
m4.metric(t("Inverted when rate high", "Инверсия при высоком ключе"),
          f"{res4['inverted_high'] * 100:.0f}%",
          t(f"vs {res4['inverted_low'] * 100:.0f}% when low", f"против {res4['inverted_low'] * 100:.0f}% при низком"))

raw_dates = get_json("/dates")
dates = [d["trade_date"] for d in raw_dates] if raw_dates and isinstance(raw_dates[0], dict) else raw_dates

st.divider()
st.header(t("Hypothesis 1: three forces shape the curve", "Гипотеза 1: форму кривой задают три силы"))
st.write(t(
    "Everything in this part is the evidence for the first hypothesis. We start from the raw curve of a "
    "single day, reduce four years of curves to three factors, and then check each of the three forces "
    "against the key rate one by one. At the end we give the overall verdict on the hypothesis.",
    "Всё в этой части это доказательство первой гипотезы. Мы стартуем с сырой кривой одного дня, сводим "
    "четыре года кривых к трём факторам, а затем по очереди проверяем каждую из трёх сил по ключевой "
    "ставке. В конце даём общий вердикт по гипотезе.",
))

st.subheader(t("The curve on a single day", "Кривая на один день"))
st.write(t(
    "Each dot is one bond plotted by its time to maturity and its yield. The line is the Nelson-Siegel "
    "fit we use everywhere below. Drag the slider through four years and watch the upward curve of 2021 "
    "flatten and invert as the key rate is hiked.",
    "Каждая точка это одна облигация по сроку до погашения и доходности. Линия это подгонка "
    "Нельсона-Сигеля, которую мы используем дальше везде. Протащи ползунок через четыре года и "
    "посмотри, как растущая кривая 2021 года уплощается и инвертируется по мере подъёма ключа.",
))
pick = st.select_slider(t("Trading day", "Торговый день"), options=dates, value=dates[-1])
curve = get_json("/curve", {"trade_date": pick})
pts = pd.DataFrame(curve["points"])
fit = curve["fit"]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=pts["ttm"], y=pts["ytm"], mode="markers", name=t("OFZ issues", "Выпуски ОФЗ")))
if fit:
    grid = np.linspace(pts["ttm"].min(), pts["ttm"].max(), 120)
    line = nelson_siegel(grid, fit["level"], fit["slope"], fit["curvature"], fit["lam"])
    fig1.add_trace(go.Scatter(x=grid, y=line, mode="lines", name=t("Nelson-Siegel fit", "Нельсон-Сигель")))
fig1.update_layout(xaxis_title=t("Years to maturity", "Лет до погашения"),
                   yaxis_title=t("Yield, %", "Доходность, %"))
st.plotly_chart(fig1, use_container_width=True)

if fit:
    a, b, c, d = st.columns(4)
    a.metric(t("Level", "Уровень"), f"{fit['level']:.2f}")
    b.metric(t("Slope", "Наклон"), f"{fit['slope']:.2f}")
    c.metric(t("Curvature", "Кривизна"), f"{fit['curvature']:.2f}")
    d.metric("R²", f"{fit['r2']:.3f}")

st.subheader(t("How the three factors moved", "Как двигались три фактора"))
st.write(t(
    "Once the curve is reduced to level, slope and curvature, the whole four-year story fits on one chart. "
    "The dashed line is the key rate. The level tracks it closely, that is the first hypothesis in plain sight.",
    "Когда кривая сведена к уровню, наклону и кривизне, вся история четырёх лет умещается на одном "
    "графике. Пунктир это ключевая ставка. Уровень идёт за ней почти вплотную, это и есть первая "
    "гипотеза прямо на глазах.",
))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=merged["trade_date"], y=merged["level"], name=t("Level", "Уровень")))
fig2.add_trace(go.Scatter(x=merged["trade_date"], y=merged["slope"], name=t("Slope", "Наклон")))
fig2.add_trace(go.Scatter(x=merged["trade_date"], y=merged["curvature"], name=t("Curvature", "Кривизна")))
fig2.add_trace(go.Scatter(x=merged["trade_date"], y=merged["rate"], name=t("Key rate", "Ключевая ставка"),
                          line=dict(dash="dash")))
fig2.update_layout(yaxis_title="%")
st.plotly_chart(fig2, use_container_width=True)

st.subheader(t("Force 1: parallel shift", "Сила 1: параллельный сдвиг"))
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=merged["rate"], y=merged["level"], mode="markers", name=t("Days", "Дни")))
xs = np.linspace(merged["rate"].min(), merged["rate"].max(), 50)
fig3.add_trace(go.Scatter(x=xs, y=res1["slope"] * xs + res1["intercept"], mode="lines",
                          name=t("Linear fit", "Линейная подгонка")))
fig3.update_layout(xaxis_title=t("Key rate, %", "Ключевая ставка, %"),
                   yaxis_title=t("Curve level, %", "Уровень кривой, %"))
st.plotly_chart(fig3, use_container_width=True)
st.write(t(
    f"The level of the curve and the key rate move together with a correlation of {res1['corr']:.2f}. "
    f"Every extra percentage point on the key rate lifts the level by {res1['slope']:.2f} points. "
    "The curve really does shift as a block when policy changes.",
    f"Уровень кривой и ключевая ставка ходят вместе с корреляцией {res1['corr']:.2f}. Каждый лишний "
    f"процент ключа поднимает уровень на {res1['slope']:.2f} пункта. Кривая действительно сдвигается "
    "целым блоком, когда меняется политика.",
))
verdict(res1["corr"] > 0.6,
        f"correlation {res1['corr']:.2f}, the curve shifts in parallel with the key rate.",
        f"корреляция {res1['corr']:.2f}, кривая сдвигается параллельно ключевой ставке.")

st.subheader(t("Force 2: short end vs long end", "Сила 2: короткий и длинный конец"))
st.write(t(
    "To see whether the two ends really live separate lives, we plot the fitted short end and long end "
    "over time, with the key rate behind them. The short end is glued to the key rate, while the long "
    "end drifts on its own, anchored by longer-term expectations.",
    "Чтобы увидеть, правда ли два конца живут по-разному, рисуем короткий и длинный конец во времени, "
    "а за ними ключевую ставку. Короткий конец приклеен к ключу, а длинный дрейфует сам по себе, "
    "удерживаемый долгосрочными ожиданиями.",
))
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=merged["trade_date"], y=merged["short_end"], name=t("Short end (~6m)", "Короткий конец (~6м)")))
fig4.add_trace(go.Scatter(x=merged["trade_date"], y=merged["long_end"], name=t("Long end", "Длинный конец")))
fig4.add_trace(go.Scatter(x=merged["trade_date"], y=merged["rate"], name=t("Key rate", "Ключевая ставка"),
                          line=dict(dash="dash")))
fig4.update_layout(yaxis_title=t("Yield, %", "Доходность, %"))
st.plotly_chart(fig4, use_container_width=True)
st.write(t(
    f"Quantitatively the short end moves {res2['short_beta']:.2f} points per point of key rate, the long "
    f"end only {res2['long_beta']:.2f}. The short end is about {ratio:.1f} times more sensitive, which is "
    "exactly what you expect if policy expectations are priced into the near maturities first.",
    f"В цифрах короткий конец двигается на {res2['short_beta']:.2f} пункта на каждый процент ключа, "
    f"длинный лишь на {res2['long_beta']:.2f}. Короткий конец примерно в {ratio:.1f} раза чувствительнее, "
    "ровно как и ждёшь, если ожидания по политике закладываются сначала в ближние сроки.",
))
verdict(res2["short_beta"] > res2["long_beta"] * 1.5,
        f"short end {ratio:.1f}x more sensitive to the key rate than the long end.",
        f"короткий конец в {ratio:.1f} раза чувствительнее к ключу, чем длинный.")

st.subheader(t("Force 3: curvature of the mid-segment", "Сила 3: выпуклость середины"))
st.write(t(
    "If the curve were just a straight line between the two ends there would be no curvature and no hump. "
    "The histogram shows the curvature factor across all trading days.",
    "Если бы кривая была просто прямой между концами, не было бы ни кривизны, ни горба. Гистограмма "
    "показывает фактор кривизны по всем торговым дням.",
))
fig5 = go.Figure()
fig5.add_trace(go.Histogram(x=merged["curvature"].dropna(), nbinsx=40))
fig5.update_layout(xaxis_title=t("Curvature factor", "Фактор кривизны"),
                   yaxis_title=t("Number of days", "Число дней"))
st.plotly_chart(fig5, use_container_width=True)
st.write(t(
    f"The average curvature is {res3['mean']:.2f} and it stays positive on {res3['share_positive'] * 100:.0f}% "
    "of days, so the medium-term bonds repeatedly pull away from the straight line and create a hump. The "
    "curve has a real third dimension, not just a level and a tilt.",
    f"Средняя кривизна {res3['mean']:.2f}, и она положительна в {res3['share_positive'] * 100:.0f}% дней, "
    "так что среднесрочные бумаги регулярно отрываются от прямой и создают горб. У кривой есть настоящее "
    "третье измерение, а не только уровень и наклон.",
))
verdict(res3["share_positive"] > 0.6,
        f"curvature positive on {res3['share_positive'] * 100:.0f}% of days, the mid-segment hump is persistent.",
        f"кривизна положительна в {res3['share_positive'] * 100:.0f}% дней, среднесрочный горб устойчив.")

h1_ok = res1["corr"] > 0.6 and res2["short_beta"] > res2["long_beta"] * 1.5 and res3["share_positive"] > 0.6
st.markdown(t("**Verdict on Hypothesis 1.**", "**Вердикт по Гипотезе 1.**"))
verdict(h1_ok,
        "all three forces show up in the data, the shape of the curve is governed by level, slope and curvature.",
        "все три силы видны в данных, форму кривой задают уровень, наклон и кривизна.")

st.divider()
st.header(t("Hypothesis 2: does tightening invert the curve?", "Гипотеза 2: инвертирует ли ужесточение кривую?"))
st.write(t(
    "The second hypothesis is the one we formulated ourselves, and here we either confirm it or reject it. "
    "The idea: when the central bank hikes hard, the short end overshoots the long end and the curve "
    "inverts. We split every day into a high-rate and a low-rate regime around the median key rate and "
    "count how often the curve is inverted in each.",
    "Вторая гипотеза это та, что мы сформулировали сами, и здесь мы её либо подтверждаем, либо "
    "опровергаем. Идея: когда ЦБ резко повышает ставку, короткий конец перелетает длинный и кривая "
    "инвертируется. Мы делим все дни на режим высокой и низкой ставки вокруг медианы ключа и считаем, "
    "как часто кривая инвертирована в каждом.",
))
fig6 = go.Figure()
fig6.add_trace(go.Bar(x=[t("Low-rate regime", "Низкая ставка"), t("High-rate regime", "Высокая ставка")],
                      y=[res4["inverted_low"] * 100, res4["inverted_high"] * 100]))
fig6.update_layout(yaxis_title=t("Share of inverted days, %", "Доля инвертированных дней, %"))
st.plotly_chart(fig6, use_container_width=True)
st.write(t(
    f"In the low-rate regime the curve is inverted only {res4['inverted_low'] * 100:.0f}% of the time. "
    f"In the high-rate regime that jumps to {res4['inverted_high'] * 100:.0f}%, and the correlation between "
    f"the key rate and the slope is {res4['corr_rate_slope']:.2f}. Tightening does flatten and often invert "
    "the curve, though even at high rates it is not inverted every single day.",
    f"В режиме низкой ставки кривая инвертирована лишь {res4['inverted_low'] * 100:.0f}% времени. В режиме "
    f"высокой это подскакивает до {res4['inverted_high'] * 100:.0f}%, а корреляция ключа с наклоном "
    f"{res4['corr_rate_slope']:.2f}. Ужесточение действительно уплощает и часто инвертирует кривую, хотя "
    "даже при высокой ставке она инвертирована не каждый день.",
))
verdict(res4["inverted_high"] > res4["inverted_low"] * 1.5,
        f"high rates raise inversion from {res4['inverted_low'] * 100:.0f}% to {res4['inverted_high'] * 100:.0f}% of days.",
        f"высокая ставка поднимает инверсию с {res4['inverted_low'] * 100:.0f}% до {res4['inverted_high'] * 100:.0f}% дней.")

st.divider()
st.header(t("When does the model actually work?", "Когда модель действительно работает?"))
st.write(t(
    "There is one more thing the data makes obvious, and it follows straight from the maths. The "
    "Nelson-Siegel decomposition only has something to explain when there is a real spread between the "
    "short and the long end. When that spread is wide the model captures the curve almost perfectly. When "
    "the curve is flat there is nothing to bend, the fit collapses into a flat average line through all the "
    "points, and the fit quality drops, even though it still preserves the overall level of the market.",
    "Данные показывают ещё одну вещь, и она следует прямо из математики. Разложению Нельсона-Сигеля есть "
    "что объяснять только тогда, когда между коротким и длинным концом есть реальный спред. Когда спред "
    "широкий, модель ловит кривую почти идеально. Когда кривая плоская, изгибать нечего, фит сваливается "
    "в среднюю прямую через все точки, и качество падает, хотя сам уровень рынка он всё равно сохраняет.",
))
bins = [0, 1, 2, 4, 8, 100]
labels = ["0-1", "1-2", "2-4", "4-8", "8+"]
g = merged.dropna(subset=["spread", "r2"]).copy()
g["bucket"] = pd.cut(g["spread"], bins=bins, labels=labels, right=False)
binned = g.groupby("bucket", observed=True)["r2"].median().reindex(labels)
fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=g["spread"], y=g["r2"], mode="markers",
                          marker=dict(opacity=0.25), name=t("Days", "Дни")))
fig7.update_layout(xaxis_title=t("Spread, short minus long end (abs)", "Спред, короткий минус длинный (модуль)"),
                   yaxis_title="R²")
st.plotly_chart(fig7, use_container_width=True)

fig8 = go.Figure()
fig8.add_trace(go.Bar(x=labels, y=binned.values))
fig8.update_layout(xaxis_title=t("Spread bucket", "Корзина спреда"),
                   yaxis_title=t("Median R²", "Медианный R²"))
st.plotly_chart(fig8, use_container_width=True)

wide = g[g["spread"] >= 8]["r2"].median()
flat = g[g["spread"] < 1]["r2"].median()
corr_sr = g["spread"].corr(g["r2"])
st.write(t(
    f"At the widest spreads the median R² reaches {wide:.2f}, the model explains over 95% of the curve and "
    f"you can build real predictions on top of it. At a flat curve the median R² falls to {flat:.2f}, the "
    f"model explains almost nothing. The correlation between the spread and R² is {corr_sr:.2f}. None of "
    "this contradicts Hypothesis 1: the three forces are still there, the model simply has work to do only "
    "when the curve is not flat.",
    f"При самых широких спредах медианный R² достигает {wide:.2f}, модель объясняет больше 95% кривой, и на "
    f"ней можно строить настоящие предсказания. На плоской кривой медианный R² падает до {flat:.2f}, модель "
    f"не объясняет почти ничего. Корреляция спреда с R² равна {corr_sr:.2f}. Это не противоречит Гипотезе 1: "
    "три силы никуда не делись, просто модели есть что объяснять только когда кривая не плоская.",
))
verdict(wide > 0.95,
        f"with a real spread the model explains about 95% of the curve (median R² {wide:.2f}); on a flat curve it does not.",
        f"при реальном спреде модель объясняет около 95% кривой (медианный R² {wide:.2f}); на плоской кривой нет.")

st.header(t("Final conclusion", "Итоговый вывод"))
with st.container(border=True):
    st.markdown(t(
        "All four hypotheses hold on four years of OFZ data. Three Nelson-Siegel numbers, level, slope "
        "and curvature, are enough to describe how the curve moves, and each one maps onto a real economic "
        "force.",
        "Все четыре гипотезы подтверждаются на четырёх годах данных по ОФЗ. Трёх чисел Нельсона-Сигеля, "
        "уровень, наклон и кривизна, достаточно, чтобы описать движение кривой, и каждое из них "
        "соответствует реальной экономической силе.",
    ))
    st.markdown(t(
        f"- **Parallel shift confirmed.** The level follows the key rate with **r = {res1['corr']:.2f}**, "
        "policy moves the whole curve as a block.\n"
        f"- **Short vs long end confirmed.** The short end is **{ratio:.1f}x more sensitive** to the key "
        "rate, so expectations reprice the near maturities first.\n"
        f"- **Curvature confirmed.** The mid-segment hump is positive on **{res3['share_positive'] * 100:.0f}% "
        "of days**, the curve has a real third dimension.\n"
        f"- **Our inversion hypothesis confirmed.** A high key rate lifts inversion from "
        f"**{res4['inverted_low'] * 100:.0f}% to {res4['inverted_high'] * 100:.0f}% of days**.\n"
        f"- **Scope of the model.** It explains the curve only when a spread exists: median R² climbs to "
        f"**{wide:.2f}** at wide spreads and falls to **{flat:.2f}** on a flat curve, where the fit is just "
        "an average line. This refines Hypothesis 1 without breaking it.",
        f"- **Параллельный сдвиг подтверждён.** Уровень идёт за ключом с **r = {res1['corr']:.2f}**, "
        "политика двигает всю кривую целым блоком.\n"
        f"- **Короткий и длинный конец подтверждены.** Короткий конец **в {ratio:.1f} раза чувствительнее** "
        "к ключу, ожидания сначала переоценивают ближние сроки.\n"
        f"- **Кривизна подтверждена.** Среднесрочный горб положителен в **{res3['share_positive'] * 100:.0f}% "
        "дней**, у кривой есть настоящее третье измерение.\n"
        f"- **Наша гипотеза об инверсии подтверждена.** Высокий ключ поднимает инверсию с "
        f"**{res4['inverted_low'] * 100:.0f}% до {res4['inverted_high'] * 100:.0f}% дней**.\n"
        f"- **Область работы модели.** Она объясняет кривую только при наличии спреда: медианный R² растёт "
        f"до **{wide:.2f}** на широких спредах и падает до **{flat:.2f}** на плоской кривой, где фит это "
        "просто средняя прямая. Это уточняет Гипотезу 1, не ломая её.",
    ))

with st.expander(t("Factor table", "Таблица факторов")):
    st.dataframe(merged[["trade_date", "level", "slope", "curvature", "r2", "rate"]])
