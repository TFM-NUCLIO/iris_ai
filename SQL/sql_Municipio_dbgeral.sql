-- SQL de extração dos dados populacionais dos municípios de 2010 até 2021.
-- Resultado dessa extração é o arquivo populacao2010_2020.csv

SELECT
	p.co_municipio_ibge, 
	m.no_municipio, 
	p.qt_populacao, 
	p.nu_ano_referencia
FROM
	dbgeral.tb_populacao p 
	INNER JOIN 
	dbgeral.tb_municipio m on
	p.co_municipio_ibge = m.co_municipio_ibge
WHERE
	p.nu_ano_referencia >2009