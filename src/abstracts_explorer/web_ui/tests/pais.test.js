/**
 * Tests for PAIS screening UI.
 */

import { jest } from '@jest/globals';

import {
    buildPaisCandidatePayload,
    closePaisCandidateModal,
    initPaisStatus,
    openPaisCandidate,
    submitPaisCandidate
} from '../static/modules/pais.js';

describe('PAIS Module', () => {
    const paper = {
        uid: 'paper-1',
        title: 'Database Paper Title',
        abstract: 'Database abstract text.',
        year: 2026,
        conference: 'PAISDB',
        url: 'https://example.test/paper'
    };

    beforeEach(() => {
        document.body.innerHTML = `
            <div id="pais-status-banner" class="hidden"></div>
            <div id="pais-modal-root"></div>
        `;
        closePaisCandidateModal();
        global.fetch.mockReset();
    });

    it('renders sanitized PAIS status without secrets', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                pais_screen_configured: true,
                pais_evidence_brief_configured: false,
                pais_extraction_configured: false,
                pais_embedding_configured: false,
                configured_models: { screen: 'screen-model' },
                configured_base_urls: { screen: 'https://example.test/v1' },
                pais_structured_output_mode: 'json_schema'
            })
        });

        await initPaisStatus();

        const banner = document.getElementById('pais-status-banner');
        expect(banner).not.toHaveClass('hidden');
        expect(banner.textContent).toContain('PAIS');
        expect(banner.textContent).toContain('Screening configured');
        expect(banner.textContent).toContain('screen-model');
        expect(banner.textContent).not.toMatch(/token|secret|api_key/i);
    });

    it('opens PAIS modal from an existing paper record', async () => {
        const event = { stopPropagation: jest.fn(), preventDefault: jest.fn() };
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => paper
        });

        await openPaisCandidate('paper-1', event);

        expect(event.stopPropagation).toHaveBeenCalled();
        expect(event.preventDefault).toHaveBeenCalled();
        expect(global.fetch).toHaveBeenCalledWith('/api/paper/paper-1');
        expect(document.getElementById('pais-article-title').textContent).toContain('Database Paper Title');
        expect(document.getElementById('pais-article-abstract').textContent).toContain('Database abstract text.');

        const editableArticleFields = Array.from(document.querySelectorAll('input, textarea'))
            .filter(element => /title|abstract|article/i.test(`${element.id} ${element.name || ''}`));
        expect(editableArticleFields).toHaveLength(0);
    });

    it('blocks submission when the selected paper has no abstract', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ ...paper, abstract: '' })
        });

        await openPaisCandidate('paper-1');

        expect(document.body.textContent).toContain('missing a title or abstract');
        expect(document.getElementById('pais-submit-button')).toBeDisabled();
    });

    it('builds PAIS payload from the selected paper and pathogen/disease metadata', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => paper
        });

        await openPaisCandidate('paper-1');
        document.getElementById('pais-pathogen-name').value = 'RSV';
        document.getElementById('pais-pathogen-synonyms').value = 'respiratory syncytial virus, hRSV';
        document.getElementById('pais-disease-name').value = 'bronchiolitis';
        document.getElementById('pais-disease-hpo-id').value = 'HP:0011950';

        const payload = buildPaisCandidatePayload(paper);

        expect(payload.article.title).toBe('Database Paper Title');
        expect(payload.article.abstract).toBe('Database abstract text.');
        expect(payload.article.publication_year).toBe(2026);
        expect(payload.article.source).toBe('PAISDB');
        expect(payload.article.source_url).toBe('https://example.test/paper');
        expect(payload.pathogen).toMatchObject({
            name: 'RSV',
            synonyms: ['respiratory syncytial virus', 'hRSV']
        });
        expect(payload.disease).toMatchObject({
            name: 'bronchiolitis',
            hpo_id: 'HP:0011950'
        });
    });

    it('submits to run-candidate and renders provenance summary fields', async () => {
        global.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => paper
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    screen_status: 'positive',
                    screen_confidence: 0.91,
                    candidate_relation_id: 10,
                    evidence_record_id: 20,
                    embedding_record_id: 30,
                    model_run_ids: [1, 2],
                    server2_called: true,
                    hosted_disagreement_flag: false,
                    quality_flags: ['ok']
                })
            });

        await openPaisCandidate('paper-1');
        document.getElementById('pais-pathogen-name').value = 'RSV';
        document.getElementById('pais-disease-name').value = 'bronchiolitis';

        await submitPaisCandidate(new Event('submit'));

        expect(global.fetch).toHaveBeenLastCalledWith(
            '/api/pais/run-candidate',
            expect.objectContaining({
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
        );
        const body = JSON.parse(global.fetch.mock.calls[1][1].body);
        expect(body.article.title).toBe('Database Paper Title');
        expect(body.article.abstract).toBe('Database abstract text.');
        expect(body.pathogen.name).toBe('RSV');
        expect(body.disease.name).toBe('bronchiolitis');

        const resultText = document.getElementById('pais-result').textContent;
        expect(resultText).toContain('screen_status');
        expect(resultText).toContain('positive');
        expect(resultText).toContain('candidate_relation_id');
        expect(resultText).toContain('10');
        expect(resultText).toContain('evidence_record_id');
        expect(resultText).toContain('20');
        expect(resultText).toContain('embedding_record_id');
        expect(resultText).toContain('30');
        expect(resultText).toContain('model_run_ids');
        expect(resultText).toContain('server2_called');
        expect(resultText).toContain('hosted_disagreement_flag');
        expect(resultText).toContain('quality_flags');
    });

    it('renders raw backend error details', async () => {
        global.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => paper
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 400,
                json: async () => ({
                    error: 'Invalid PAIS candidate',
                    details: [{ loc: ['pathogen', 'name'], msg: 'required' }]
                })
            });

        await openPaisCandidate('paper-1');
        document.getElementById('pais-pathogen-name').value = 'RSV';
        document.getElementById('pais-disease-name').value = 'bronchiolitis';

        await submitPaisCandidate(new Event('submit'));

        const resultHtml = document.getElementById('pais-result').innerHTML;
        expect(resultHtml).toContain('Invalid PAIS candidate');
        expect(resultHtml).toContain('Raw error details');
        expect(resultHtml).toContain('pathogen');
    });
});
