"""Configuracion centralizada de indicadores por fuente."""

WHO_INDICATORS = {
    "NCDMORT3070": {
        "indicator_name": "Probabilidad de morir entre 30 y 70 anos por ENT",
        "role": "target",
    },
    "WHOSIS_000001": {
        "indicator_name": "Esperanza de vida al nacer",
        "role": "descriptive",
    },
    "MDG_0000000001": {
        "indicator_name": "Tasa de mortalidad infantil",
        "role": "descriptive",
    },
    "WHS4_100": {
        "indicator_name": "Camas hospitalarias por 10,000 habitantes",
        "role": "health_feature",
    },
    "TB_e_inc_num": {
        "indicator_name": "Incidencia de tuberculosis",
        "role": "descriptive",
    },
}

WORLD_BANK_INDICATORS = {
    "SH.XPD.CHEX.GD.ZS": {
        "indicator_name": "Gasto en salud como % del PIB",
        "role": "predictive_feature",
    },
    "SH.MED.PHYS.ZS": {
        "indicator_name": "Medicos por 1,000 habitantes",
        "role": "health_feature",
    },
    "SP.DYN.LE00.IN": {
        "indicator_name": "Esperanza de vida",
        "role": "descriptive",
    },
    "SH.H2O.BASW.ZS": {
        "indicator_name": "Acceso a agua potable",
        "role": "health_social_feature",
    },
    "NY.GDP.PCAP.CD": {
        "indicator_name": "PIB per capita en USD",
        "role": "socioeconomic_feature",
    },
}
