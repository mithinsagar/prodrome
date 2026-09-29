{#
  Resolve which run the models read.

  Every fact table is keyed by run_id, so a model that forgot to filter would
  silently union every historical run and double-count everything. Centralising
  the resolution here means a model can only get it right: it calls
  `{{ current_run() }}` and cannot express "all runs".

  "latest" resolves to the most recent *finished* run, so an in-flight or crashed
  run is never published as though it were complete.
#}
{% macro current_run() %}
    {%- if var('run_id') == 'latest' -%}
        (SELECT run_id FROM {{ source('prodrome', 'runs') }}
         WHERE finished_at IS NOT NULL
         ORDER BY finished_at DESC LIMIT 1)
    {%- else -%}
        '{{ var('run_id') }}'
    {%- endif -%}
{% endmacro %}


{#
  Quarter label ("2023Q3") to a sortable integer (20233), and to a quarter index
  for arithmetic. Written as macros because they appear in several models and a
  divergent copy would silently misorder a series.
#}
{% macro quarter_sort_key(column) %}
    CAST(substr({{ column }}, 1, 4) AS INTEGER) * 4
      + CAST(substr({{ column }}, 6, 1) AS INTEGER) - 1
{% endmacro %}


{% macro quarter_label_to_date(column) %}
    make_date(
        CAST(substr({{ column }}, 1, 4) AS INTEGER),
        CAST(substr({{ column }}, 6, 1) AS INTEGER) * 3,
        1
    )
{% endmacro %}
