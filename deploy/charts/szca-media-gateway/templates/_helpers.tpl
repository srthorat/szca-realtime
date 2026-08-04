{{- define "szca-media-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "szca-media-gateway.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "szca-media-gateway.labels" -}}
helm.sh/chart: {{ include "szca-media-gateway.name" . }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "szca-media-gateway.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "szca-media-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "szca-media-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
