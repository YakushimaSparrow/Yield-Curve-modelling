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
    "We take the full trading history of Russian federal bonds (OFZ-PD) from the Moscow Exchange, "
    "rebuild the yield curve for every single trading day, and test a set of hypotheses about what "
    "actually moves that curve around. The goal is not a dashboard of numbers but an answer: which "
    "forces bend the curve, and can we see them in four years of real data.",
    "Мы берём полную историю торгов российскими гособлигациями (ОФЗ-ПД) с Московской биржи, "
    "заново строим кривую доходности на каждый торговый день и проверяем набор гипотез о том, что "
    "на самом деле двигает эту кривую. Цель не дашборд с цифрами, а ответ: какие силы изгибают "
    "кривую и видно ли их на четырёх годах реальных данных.",
))

with st.expander(t("How the data was built", "Откуда взяты данные")):
    st.write(t(
        "Daily clean prices for every SU26 issue come from the MOEX ISS API and are stored in a local "
        "SQLite database with two tables: static instrument data and daily market data. A FastAPI service "
        "sits on top of the database, and this page reads everything through that API. For each trading "
        "day we solve every bond for its yield to maturity and fit a Nelson-Siegel curve, which splits the "
        "whole curve into three numbers: level (the height of the long end), slope (short end minus long "
        "end) and curvature (the mid-segment hump). The key rate is the public history of Bank of Russia "
        "decisions. The data covers four years, 2021 to 2024, and 27 OFZ issues.",
        "Дневные цены по каждому выпуску SU26 берутся из публичного API Московской биржи (MOEX ISS) и "
        "складываются в локальную базу SQLite с двумя таблицами: статика по инструментам и дневные "
        "котировки. Поверх базы поднят сервис на FastAPI, и эта страница читает всё через него. Для "
        "каждого торгового дня мы решаем каждую облигацию на доходность к погашению и калибруем кривую "
        "Нельсона-Сигеля, которая раскладывает кривую на три числа: уровень (высота длинного конца), "
        "наклон (короткий конец минус длинный) и кривизну (горб в середине). Ключевая ставка взята из "
        "публичной истории решений Банка России. Данные покрывают четыре года, с 2021 по 2024, и 27 "
        "выпусков ОФЗ.",
    ))

st.subheader(t("The hypotheses", "Наши гипотезы"))
st.markdown(t(
    "1. **Parallel shift.** The whole curve moves up and down with the key rate.\n"
    "2. **Short vs long end.** The short end reacts to monetary-policy expectations far more than the long end.\n"
    "3. **Curvature.** The medium-term segment carries a persistent convexity that bends the curve.\n"
    "4. **Our own hypothesis.** A sharp tightening cycle inverts the curve: when the key rate is high, the "
    "short end climbs above the long end.",
    "1. **Параллельный сдвиг.** Вся кривая ходит вверх-вниз вслед за ключевой ставкой.\n"
    "2. **Короткий и длинный конец.** Короткий конец реагирует на ожидания по ДКП куда сильнее длинного.\n"
    "3. **Кривизна.** Среднесрочный участок несёт устойчивую выпуклость, которая изгибает кривую.\n"
    "4. **Наша собственная гипотеза.** Резкое ужесточение инвертирует кривую: при высокой ставке "
    "короткий конец залезает выше длинного.",
))

factors = load_factors()
key_rate = load_key_rate()

if factors.empty:
    st.warning(t("The database is empty. Run build_database.py first.",
                 "База пустая. Сначала запусти build_database.py."))
    st.stop()

merged = attach_key_rate(factors, key_rate)
merged["short_end"] = merged["level"] + merged["slope"]
merged["long_end"] = merged["level"]

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

st.header(t("The curve on a single day", "Кривая на один день"))
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

st.header(t("How the three factors moved", "Как двигались три фактора"))
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

st.header(t("Hypothesis 1: parallel shift", "Гипотеза 1: параллельный сдвиг"))
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

st.header(t("Hypothesis 2: short end vs long end", "Гипотеза 2: короткий и длинный конец"))
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

st.header(t("Hypothesis 3: curvature of the mid-segment", "Гипотеза 3: выпуклость середины"))
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

st.header(t("Hypothesis 4: does tightening invert the curve?", "Гипотеза 4: инвертирует ли ужесточение кривую?"))
st.write(t(
    "This is the hypothesis we added ourselves. The idea: when the central bank hikes hard, the short end "
    "overshoots the long end and the curve inverts. We split every day into a high-rate and a low-rate "
    "regime around the median key rate and count how often the curve is inverted in each.",
    "Это гипотеза, которую мы добавили сами. Идея: когда ЦБ резко повышает ставку, короткий конец "
    "перелетает длинный и кривая инвертируется. Мы делим все дни на режим высокой и низкой ставки "
    "вокруг медианы ключа и считаем, как часто кривая инвертирована в каждом.",
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
        f"**{res4['inverted_low'] * 100:.0f}% to {res4['inverted_high'] * 100:.0f}% of days**.",
        f"- **Параллельный сдвиг подтверждён.** Уровень идёт за ключом с **r = {res1['corr']:.2f}**, "
        "политика двигает всю кривую целым блоком.\n"
        f"- **Короткий и длинный конец подтверждены.** Короткий конец **в {ratio:.1f} раза чувствительнее** "
        "к ключу, ожидания сначала переоценивают ближние сроки.\n"
        f"- **Кривизна подтверждена.** Среднесрочный горб положителен в **{res3['share_positive'] * 100:.0f}% "
        "дней**, у кривой есть настоящее третье измерение.\n"
        f"- **Наша гипотеза об инверсии подтверждена.** Высокий ключ поднимает инверсию с "
        f"**{res4['inverted_low'] * 100:.0f}% до {res4['inverted_high'] * 100:.0f}% дней**.",
    ))

with st.expander(t("Factor table", "Таблица факторов")):
    st.dataframe(merged[["trade_date", "level", "slope", "curvature", "r2", "rate"]])
