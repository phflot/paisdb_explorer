/**
 * PAIS screening module.
 *
 * PAIS screening is intentionally anchored to existing database records. The UI
 * never asks users to paste an article title or abstract; those fields are
 * sourced from /api/paper/<uid> and shown only as read-only provenance.
 */

import { API_BASE } from './utils/constants.js';
import { escapeHtml } from './utils/dom-utils.js';

let currentPaisPaper = null;
let activePaisRequestId = 0;

function asText(value) {
    return value === undefined || value === null ? '' : String(value).trim();
}

function optionalText(value) {
    const text = asText(value);
    return text || null;
}

function commaList(value) {
    return asText(value)
        .split(',')
        .map(item => item.trim())
        .filter(Boolean);
}

function putIfPresent(target, key, value) {
    if (value !== null && value !== undefined && value !== '') {
        target[key] = value;
    }
}

function getFormValue(id) {
    const element = document.getElementById(id);
    return element ? element.value : '';
}

function getPaisRoot() {
    let root = document.getElementById('pais-modal-root');
    if (!root) {
        root = document.createElement('div');
        root.id = 'pais-modal-root';
        document.body.appendChild(root);
    }
    return root;
}

function paperSourceUrl(paper) {
    return optionalText(paper.url) || optionalText(paper.paper_pdf_url);
}

function paperPublicationYear(paper) {
    const year = Number.parseInt(paper.year, 10);
    return Number.isInteger(year) ? year : null;
}

function hasScreenableArticle(paper) {
    return Boolean(optionalText(paper.title) && optionalText(paper.abstract));
}

function renderJson(value) {
    return escapeHtml(JSON.stringify(value, null, 2));
}

function renderValue(value) {
    if (Array.isArray(value)) {
        return value.length ? value.map(item => escapeHtml(String(item))).join(', ') : '<span class="pais-muted">None</span>';
    }
    if (value === true) return '<span class="pais-badge pais-badge-success">true</span>';
    if (value === false) return '<span class="pais-badge pais-badge-neutral">false</span>';
    if (value === null || value === undefined || value === '') return '<span class="pais-muted">None</span>';
    return escapeHtml(String(value));
}

function renderResultRow(label, value) {
    return `
        <div class="pais-result-row">
            <dt>${escapeHtml(label)}</dt>
            <dd>${renderValue(value)}</dd>
        </div>
    `;
}

function renderQualityFlags(flags) {
    if (!Array.isArray(flags) || flags.length === 0) {
        return '<span class="pais-muted">None</span>';
    }
    return flags.map(flag => `<span class="pais-badge pais-badge-warning">${escapeHtml(String(flag))}</span>`).join(' ');
}

function renderModelRunIds(ids) {
    if (!Array.isArray(ids) || ids.length === 0) {
        return '<span class="pais-muted">None</span>';
    }
    return ids.map(id => `<span class="pais-badge pais-badge-neutral">${escapeHtml(String(id))}</span>`).join(' ');
}

function renderPaisResult(result) {
    const resultEl = document.getElementById('pais-result');
    if (!resultEl) return;

    resultEl.innerHTML = `
        <div class="pais-result-card">
            <div class="pais-result-card-header">
                <span class="pais-badge pais-badge-success">PAIS result</span>
                <span class="pais-muted">Persisted screening summary</span>
            </div>
            <dl class="pais-result-grid">
                ${renderResultRow('screen_status', result.screen_status)}
                ${renderResultRow('screen_confidence', result.screen_confidence)}
                ${renderResultRow('candidate_relation_id', result.candidate_relation_id)}
                ${renderResultRow('evidence_record_id', result.evidence_record_id)}
                ${renderResultRow('embedding_record_id', result.embedding_record_id)}
                <div class="pais-result-row">
                    <dt>model_run_ids</dt>
                    <dd>${renderModelRunIds(result.model_run_ids)}</dd>
                </div>
                ${renderResultRow('server2_called', result.server2_called)}
                ${renderResultRow('hosted_disagreement_flag', result.hosted_disagreement_flag)}
                <div class="pais-result-row">
                    <dt>quality_flags</dt>
                    <dd>${renderQualityFlags(result.quality_flags)}</dd>
                </div>
            </dl>
        </div>
    `;
}

function renderPaisError(message, details = null) {
    const resultEl = document.getElementById('pais-result');
    if (!resultEl) return;

    resultEl.innerHTML = `
        <div class="pais-error-card">
            <div class="pais-result-card-header">
                <span class="pais-badge pais-badge-danger">PAIS error</span>
                <span>${escapeHtml(message)}</span>
            </div>
            ${details ? `
                <details class="pais-error-details">
                    <summary>Raw error details</summary>
                    <pre>${renderJson(details)}</pre>
                </details>
            ` : ''}
        </div>
    `;
}

function renderStatusUnavailable(errorMessage) {
    const banner = document.getElementById('pais-status-banner');
    if (!banner) return;
    banner.classList.remove('hidden');
    banner.innerHTML = `
        <div class="pais-status pais-status-warning">
            <span class="pais-badge pais-badge-warning">PAIS</span>
            <span>Configuration status unavailable</span>
            <span class="pais-muted">${escapeHtml(errorMessage)}</span>
        </div>
    `;
}

function renderPaisStatus(status) {
    const banner = document.getElementById('pais-status-banner');
    if (!banner) return;

    const screenReady = Boolean(status.pais_screen_configured);
    const configuredStages = [
        status.pais_screen_configured,
        status.pais_evidence_brief_configured,
        status.pais_extraction_configured,
        status.pais_embedding_configured
    ].filter(Boolean).length;
    const models = status.configured_models || {};
    const modelText = Object.entries(models)
        .filter(([, value]) => value)
        .map(([key, value]) => `${key}: ${value}`)
        .join(' · ');

    banner.classList.remove('hidden');
    banner.innerHTML = `
        <div class="pais-status ${screenReady ? 'pais-status-ready' : 'pais-status-warning'}">
            <span class="pais-badge ${screenReady ? 'pais-badge-success' : 'pais-badge-warning'}">PAIS</span>
            <span>${screenReady ? 'Screening configured' : 'Screening not configured'}</span>
            <span class="pais-muted">${configuredStages}/4 stages configured</span>
            ${modelText ? `<span class="pais-status-models">${escapeHtml(modelText)}</span>` : ''}
        </div>
    `;
}

function setSubmitPending(isPending) {
    const submit = document.getElementById('pais-submit-button');
    if (!submit) return;
    submit.disabled = isPending || submit.dataset.blocked === 'true';
    submit.innerHTML = isPending
        ? '<i class="fas fa-spinner fa-spin"></i><span>Screening...</span>'
        : '<i class="fas fa-microscope"></i><span>Run PAIS screen</span>';
}

function renderPaisModalLoading() {
    const root = getPaisRoot();
    root.innerHTML = `
        <div class="pais-modal-backdrop" onclick="if (event.target === this) closePaisCandidateModal()">
            <div class="pais-modal-panel pais-modal-panel-narrow">
                <div class="pais-modal-header">
                    <div>
                        <span class="pais-badge pais-badge-neutral">PAIS</span>
                        <h2>Loading paper record</h2>
                    </div>
                    <button type="button" class="pais-icon-button" onclick="closePaisCandidateModal()" aria-label="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="pais-loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>Fetching full paper details...</span>
                </div>
            </div>
        </div>
    `;
}

function renderPaisModalError(message) {
    const root = getPaisRoot();
    root.innerHTML = `
        <div class="pais-modal-backdrop" onclick="if (event.target === this) closePaisCandidateModal()">
            <div class="pais-modal-panel pais-modal-panel-narrow">
                <div class="pais-modal-header">
                    <div>
                        <span class="pais-badge pais-badge-danger">PAIS</span>
                        <h2>Cannot open PAIS screen</h2>
                    </div>
                    <button type="button" class="pais-icon-button" onclick="closePaisCandidateModal()" aria-label="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="pais-error-card">
                    ${escapeHtml(message)}
                </div>
            </div>
        </div>
    `;
}

function renderPaisModal(paper) {
    const root = getPaisRoot();
    const title = optionalText(paper.title) || 'Untitled';
    const abstract = optionalText(paper.abstract);
    const sourceUrl = paperSourceUrl(paper);
    const screenable = hasScreenableArticle(paper);
    const publicationYear = paperPublicationYear(paper);

    root.innerHTML = `
        <div class="pais-modal-backdrop" onclick="if (event.target === this) closePaisCandidateModal()">
            <div class="pais-modal-panel">
                <div class="pais-modal-header">
                    <div>
                        <span class="pais-badge pais-badge-success">PAIS candidate</span>
                        <h2>Screen database paper</h2>
                        <p>Title and abstract are sourced from the selected database record.</p>
                    </div>
                    <button type="button" class="pais-icon-button" onclick="closePaisCandidateModal()" aria-label="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>

                <div class="pais-provenance">
                    <div class="pais-section-heading">Selected record provenance</div>
                    <h3 id="pais-article-title">${escapeHtml(title)}</h3>
                    <div class="pais-provenance-meta">
                        <span>UID: ${escapeHtml(asText(paper.uid) || 'unknown')}</span>
                        ${publicationYear ? `<span>Year: ${publicationYear}</span>` : ''}
                        ${paper.conference ? `<span>Source: ${escapeHtml(paper.conference)}</span>` : ''}
                        ${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">Source URL</a>` : ''}
                    </div>
                    <div id="pais-article-abstract" class="pais-provenance-abstract">
                        ${abstract ? escapeHtml(abstract) : '<span class="pais-muted">No abstract available</span>'}
                    </div>
                </div>

                ${!screenable ? `
                    <div class="pais-error-card">
                        This database record is missing a title or abstract. PAIS screening is blocked until the built database contains the required article text.
                    </div>
                ` : ''}

                <form id="pais-candidate-form" class="pais-form" onsubmit="submitPaisCandidate(event)">
                    <div class="pais-form-grid">
                        <fieldset>
                            <legend>Pathogen</legend>
                            <label for="pais-pathogen-name">Name <span aria-hidden="true">*</span></label>
                            <input id="pais-pathogen-name" required autocomplete="off" placeholder="e.g. RSV">

                            <label for="pais-pathogen-normalized-name">Normalized name</label>
                            <input id="pais-pathogen-normalized-name" autocomplete="off" placeholder="optional">

                            <label for="pais-pathogen-synonyms">Synonyms</label>
                            <input id="pais-pathogen-synonyms" autocomplete="off" placeholder="comma-separated, optional">

                            <div class="pais-form-grid-compact">
                                <div>
                                    <label for="pais-pathogen-ncbi-taxid">NCBI taxid</label>
                                    <input id="pais-pathogen-ncbi-taxid" autocomplete="off" placeholder="optional">
                                </div>
                                <div>
                                    <label for="pais-pathogen-taxonomic-rank">Taxonomic rank</label>
                                    <input id="pais-pathogen-taxonomic-rank" autocomplete="off" placeholder="optional">
                                </div>
                            </div>

                            <label for="pais-pathogen-strain-or-variant">Strain or variant</label>
                            <input id="pais-pathogen-strain-or-variant" autocomplete="off" placeholder="optional">
                        </fieldset>

                        <fieldset>
                            <legend>Disease / phenotype</legend>
                            <label for="pais-disease-name">Name <span aria-hidden="true">*</span></label>
                            <input id="pais-disease-name" required autocomplete="off" placeholder="e.g. bronchiolitis">

                            <label for="pais-disease-normalized-name">Normalized name</label>
                            <input id="pais-disease-normalized-name" autocomplete="off" placeholder="optional">

                            <label for="pais-disease-synonyms">Synonyms</label>
                            <input id="pais-disease-synonyms" autocomplete="off" placeholder="comma-separated, optional">

                            <div class="pais-form-grid-compact">
                                <div>
                                    <label for="pais-disease-doid">DOID</label>
                                    <input id="pais-disease-doid" autocomplete="off" placeholder="optional">
                                </div>
                                <div>
                                    <label for="pais-disease-hpo-id">HPO ID</label>
                                    <input id="pais-disease-hpo-id" autocomplete="off" placeholder="optional">
                                </div>
                                <div>
                                    <label for="pais-disease-mondo-id">MONDO ID</label>
                                    <input id="pais-disease-mondo-id" autocomplete="off" placeholder="optional">
                                </div>
                            </div>
                        </fieldset>
                    </div>

                    <div class="pais-modal-actions">
                        <button type="button" class="pais-button pais-button-secondary" onclick="closePaisCandidateModal()">
                            Cancel
                        </button>
                        <button
                            id="pais-submit-button"
                            type="submit"
                            class="pais-button pais-button-primary"
                            data-blocked="${screenable ? 'false' : 'true'}"
                            ${screenable ? '' : 'disabled'}
                        >
                            <i class="fas fa-microscope"></i><span>Run PAIS screen</span>
                        </button>
                    </div>
                </form>

                <div id="pais-result" class="pais-result"></div>
            </div>
        </div>
    `;
}

export function buildPaisCandidatePayload(paper) {
    const title = optionalText(paper.title);
    const abstract = optionalText(paper.abstract);
    if (!title || !abstract) {
        throw new Error('Selected paper is missing a title or abstract.');
    }

    const article = {
        title,
        abstract,
        source: optionalText(paper.conference) || 'abstracts_explorer'
    };
    putIfPresent(article, 'publication_year', paperPublicationYear(paper));
    putIfPresent(article, 'source_url', paperSourceUrl(paper));

    const pathogen = {
        name: asText(getFormValue('pais-pathogen-name'))
    };
    putIfPresent(pathogen, 'normalized_name', optionalText(getFormValue('pais-pathogen-normalized-name')));
    const pathogenSynonyms = commaList(getFormValue('pais-pathogen-synonyms'));
    if (pathogenSynonyms.length) pathogen.synonyms = pathogenSynonyms;
    putIfPresent(pathogen, 'ncbi_taxid', optionalText(getFormValue('pais-pathogen-ncbi-taxid')));
    putIfPresent(pathogen, 'taxonomic_rank', optionalText(getFormValue('pais-pathogen-taxonomic-rank')));
    putIfPresent(pathogen, 'strain_or_variant', optionalText(getFormValue('pais-pathogen-strain-or-variant')));

    const disease = {
        name: asText(getFormValue('pais-disease-name'))
    };
    putIfPresent(disease, 'normalized_name', optionalText(getFormValue('pais-disease-normalized-name')));
    const diseaseSynonyms = commaList(getFormValue('pais-disease-synonyms'));
    if (diseaseSynonyms.length) disease.synonyms = diseaseSynonyms;
    putIfPresent(disease, 'doid', optionalText(getFormValue('pais-disease-doid')));
    putIfPresent(disease, 'hpo_id', optionalText(getFormValue('pais-disease-hpo-id')));
    putIfPresent(disease, 'mondo_id', optionalText(getFormValue('pais-disease-mondo-id')));

    return { article, pathogen, disease };
}

export async function initPaisStatus() {
    const banner = document.getElementById('pais-status-banner');
    if (!banner) return;

    try {
        const response = await fetch(`${API_BASE}/api/pais/status`);
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.error) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }
        renderPaisStatus(data);
    } catch (error) {
        renderStatusUnavailable(error.message || 'Unknown error');
    }
}

export async function openPaisCandidate(paperUid, event = null) {
    if (event) {
        event.stopPropagation();
        if (event.preventDefault) event.preventDefault();
    }

    const uid = asText(paperUid);
    if (!uid) {
        renderPaisModalError('Cannot screen this paper because it has no database UID.');
        return;
    }

    renderPaisModalLoading();
    try {
        const response = await fetch(`${API_BASE}/api/paper/${encodeURIComponent(uid)}`);
        const paper = await response.json().catch(() => ({}));
        if (!response.ok || paper.error) {
            throw new Error(paper.error || `HTTP ${response.status}`);
        }
        currentPaisPaper = paper;
        renderPaisModal(paper);
    } catch (error) {
        currentPaisPaper = null;
        renderPaisModalError(`Failed to load paper details: ${error.message}`);
    }
}

export function closePaisCandidateModal() {
    const root = document.getElementById('pais-modal-root');
    if (root) {
        root.innerHTML = '';
    }
    currentPaisPaper = null;
}

export async function submitPaisCandidate(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    if (!currentPaisPaper) {
        renderPaisError('No selected paper is loaded.');
        return;
    }

    let payload;
    try {
        payload = buildPaisCandidatePayload(currentPaisPaper);
    } catch (error) {
        renderPaisError(error.message);
        return;
    }

    const requestId = ++activePaisRequestId;
    setSubmitPending(true);
    renderPaisResult({
        screen_status: 'running',
        screen_confidence: null,
        candidate_relation_id: null,
        evidence_record_id: null,
        embedding_record_id: null,
        model_run_ids: [],
        server2_called: null,
        hosted_disagreement_flag: null,
        quality_flags: []
    });

    try {
        const response = await fetch(`${API_BASE}/api/pais/run-candidate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json().catch(() => ({}));
        if (requestId !== activePaisRequestId) return;
        if (!response.ok || data.error) {
            renderPaisError(data.error || `HTTP ${response.status}`, data);
            return;
        }
        renderPaisResult(data);
    } catch (error) {
        if (requestId === activePaisRequestId) {
            renderPaisError(error.message || 'PAIS request failed');
        }
    } finally {
        if (requestId === activePaisRequestId) {
            setSubmitPending(false);
        }
    }
}
