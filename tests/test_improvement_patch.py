
"""Regression/negative-path checks for the improvement patch."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from net_hdbscan import hdbscan, fit_hierarchy, extract_clusters, compare_extractions
from net_hdbscan import pipeline
from net_hdbscan.network import NeighborPairLimitError
from net_hdbscan.diagnostics import group_diagnostics


def _distances():
    x = np.array([0, 1, 2, 10, 11, 12], dtype=float)
    d = np.abs(x[:,None]-x[None,:])
    i,j=np.where(~np.eye(len(x), dtype=bool))
    return csr_matrix((d[i,j],(i,j)), shape=d.shape)


@pytest.mark.parametrize('method', ['eom','leaf'])
def test_reusable_fit_matches_original_public_api(method):
    graph=_distances()
    fit=fit_hierarchy(graph,min_cluster_size=2,min_samples=2)
    a=extract_clusters(fit,method=method)
    b=hdbscan(graph,min_cluster_size=2,min_samples=2,cluster_selection_method=method)
    np.testing.assert_array_equal(a.labels,b.labels)
    np.testing.assert_allclose(a.probabilities,b.probabilities)
    np.testing.assert_allclose(a.core_distances,b.core_distances)
    assert not fit.core_distances.flags.writeable
    assert not fit.tree.parent.flags.writeable


def test_compare_extractions_no_refit():
    fitted=fit_hierarchy(_distances(),min_cluster_size=2)
    compared=compare_extractions(fitted)
    assert set(compared.results)=={'eom','leaf'}
    assert len(compared.summary)==2
    assert all(row['retained_share'] <= 1.0 for row in compared.summary)


def test_invalid_extraction_does_not_poison_fitted_hierarchy():
    fit=fit_hierarchy(_distances(),min_cluster_size=2)
    with pytest.raises(ValueError):
        extract_clusters(fit,method='invented')
    extract_clusters(fit,method='leaf')


def test_basic_diagnostics_structural_vs_truncation():
    result=group_diagnostics(
        n_observations=4, position=np.arange(4), weights=np.ones(4,dtype=np.int64),
        position_components=np.array([0,0,0,1]),labels=np.array([-1,-1,-1,-1]),
        core_distances=np.array([1,np.inf,2,np.inf]), cluster_stability=np.zeros(1),
        cluster_selected=np.zeros(1,dtype=bool), effective_min_samples=2,
        min_cluster_size=3, allow_single_cluster=False)
    assert result['n_observations_structurally_unreachable']==1
    assert result['n_observations_core_truncated']==1
    assert result['n_observations_core_unresolved']==2
    assert result['selected_cluster_count']==0


def test_adaptive_later_pair_limit_is_fatal(monkeypatch):
    # Tests the caller's error behavior using the actual adaptive function.
    from net_hdbscan import HDBSCANConfig
    from net_hdbscan.pipeline import _cluster_unit_with_distance_search
    from net_hdbscan.network import Snaps
    import geopandas as gpd
    from shapely.geometry import Point
    points=gpd.GeoDataFrame(geometry=[Point(0,0),Point(1,0)], crs='EPSG:3857')
    # The fake first run returns a valid empty selection, so a second trial is attempted.
    def fake_cluster(points,snaps,graph,config):
        if config.max_distance>2:
            raise NeighborPairLimitError(11,10,config.max_distance)
        return points, None, None, {'share_core_truncated':1.0, 'share_core_structurally_unreachable':0.0,
                                     'n_clusters':0,'noise_share':1.0, 'n_pairs':0}
    monkeypatch.setattr(pipeline,'_cluster_unit',fake_cluster)
    cfg=HDBSCANConfig(max_distance=8, min_distance=2, distance_mode='adaptive', distance_growth=2)
    with pytest.raises(NeighborPairLimitError):
        _cluster_unit_with_distance_search(points,object(),object(),cfg)

def test_adaptive_first_pair_limit_is_fatal(monkeypatch):
    from net_hdbscan import HDBSCANConfig
    import geopandas as gpd
    from shapely.geometry import Point
    points=gpd.GeoDataFrame(geometry=[Point(0,0),Point(1,0)],crs='EPSG:3857')
    def fails(*args):
        raise NeighborPairLimitError(11,10,2.0)
    monkeypatch.setattr(pipeline,'_cluster_unit',fails)
    cfg=HDBSCANConfig(max_distance=8,min_distance=2,distance_mode='adaptive')
    with pytest.raises(NeighborPairLimitError):
        pipeline._cluster_unit_with_distance_search(points,object(),object(),cfg)


def test_publication_failure_preserves_prior_generation(files, tmp_path, monkeypatch):
    import hashlib
    import net_hdbscan.pipeline as pipe
    from net_hdbscan import HDBSCANConfig, cluster_files
    cfg=HDBSCANConfig(max_distance=600,min_cluster_size=15)
    output=tmp_path/'run'
    kwargs=dict(points_path=files['points'],network_path=files['roads'],
                output_dir=output,config=cfg)
    cluster_files(**kwargs)
    before={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob('*') if p.is_file()}
    def fail_csv(*args,**kwargs):
        raise OSError('injected summary write error')
    monkeypatch.setattr(pipe,'write_csv',fail_csv)
    with pytest.raises(OSError,match='injected'):
        cluster_files(**kwargs,force=True)
    after={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in output.rglob('*') if p.is_file()}
    assert before==after
    assert not list(output.parent.glob('.run.stage-*'))


@pytest.mark.parametrize('grouped', [False, True])
def test_publication_rejects_missing_parquet_support_before_hashing(monkeypatch, tmp_path, grouped):
    # Ensure the transaction wrapper preserves the original early validation
    # even when the input files do not exist.  The old wrapper hashed first.
    from net_hdbscan import HDBSCANConfig, cluster_files, cluster_files_by_column
    import net_hdbscan.pipeline as pipe
    import net_hdbscan.publication as pub
    calls = []

    def no_parquet():
        calls.append('pyarrow')
        raise ImportError('simulated missing pyarrow')

    def no_hash(_):
        calls.append('hash')
        raise AssertionError('input hashing preceded dependency validation')

    monkeypatch.setattr(pipe, '_require_pyarrow', no_parquet)
    monkeypatch.setattr(pub, '_input_checksums', no_hash)
    kwargs = dict(points_path='nonexistent-points.gpkg', network_path='nonexistent-network.gpkg',
                  output_dir=tmp_path/'out', config=HDBSCANConfig(max_distance=100))
    if grouped:
        kwargs['group_col'] = 'group'
    with pytest.raises(ImportError, match='simulated missing pyarrow'):
        (cluster_files_by_column if grouped else cluster_files)(**kwargs)
    assert calls == ['pyarrow']


def test_publication_manifest_integrity(files,tmp_path):
    import hashlib, json
    from net_hdbscan import HDBSCANConfig, cluster_files
    output=tmp_path/'verify'
    cluster_files(points_path=files['points'],network_path=files['roads'],
                  output_dir=output,config=HDBSCANConfig(max_distance=600,min_cluster_size=15))
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['manifest_schema_version']==2
    for path,info in manifest['output_integrity'].items():
        assert hashlib.sha256((output/path).read_bytes()).hexdigest()==info['sha256']


def test_group_failure_stops_following_groups(points, roads, tmp_path, monkeypatch):
    import geopandas as gpd
    from net_hdbscan import HDBSCANConfig, cluster_files_by_column
    data=points.copy()
    data['sector']=np.where(np.arange(len(data)) < len(data)//2,'a','b')
    p=tmp_path/'p.parquet'; n=tmp_path/'n.parquet'; out=tmp_path/'out'
    data.to_parquet(p); roads.to_parquet(n)
    real= pipeline._run_unit
    called=[]
    def broken(*args,**kwargs):
        called.append(len(called)+1)
        if len(called)==2:
            raise RuntimeError('injected group failure')
        return real(*args,**kwargs)
    monkeypatch.setattr(pipeline,'_run_unit',broken)
    with pytest.raises(RuntimeError,match='injected group failure'):
        cluster_files_by_column(points_path=p,network_path=n,output_dir=out,
                                group_col='sector',config=HDBSCANConfig(max_distance=600,min_cluster_size=15))
    assert called == [1,2]
    assert not out.exists()
