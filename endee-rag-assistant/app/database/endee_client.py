from __future__ import annotations

import json
from typing import Any, Dict, List

import msgpack
import requests


class EndeeClientError(RuntimeError):
    pass


class EndeeClient:
    def __init__(self, base_url: str, auth_token: str = "", timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected_statuses: tuple[int, ...] = (200,),
        json_body: Dict[str, Any] | List[Dict[str, Any]] | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = self._headers()

        payload = None
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(json_body)

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise EndeeClientError(f"Failed to connect to Endee at {url}: {exc}") from exc

        if response.status_code not in expected_statuses:
            detail = response.text.strip()
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    detail = str(parsed.get("error") or parsed.get("detail") or detail)
            except ValueError:
                pass

            raise EndeeClientError(
                f"Endee request failed ({method} {path}) with status "
                f"{response.status_code}: {detail}"
            )

        return response

    def health_check(self) -> bool:
        try:
            self._request("GET", "/api/v1/health")
        except EndeeClientError:
            return False
        return True

    def list_indexes(self) -> List[Dict[str, Any]]:
        response = self._request("GET", "/api/v1/index/list")
        payload = response.json()
        indexes = payload.get("indexes", [])
        if not isinstance(indexes, list):
            return []
        return indexes

    def create_index(
        self,
        index_name: str,
        dim: int,
        space_type: str = "cosine",
        precision: str = "float32",
    ) -> None:
        response = self._request(
            "POST",
            "/api/v1/index/create",
            expected_statuses=(200, 409),
            json_body={
                "index_name": index_name,
                "dim": dim,
                "space_type": space_type,
                "precision": precision,
            },
        )

        if response.status_code == 409:
            if "already" in response.text.lower() and "exist" in response.text.lower():
                return
            raise EndeeClientError(
                f"Could not create Endee index '{index_name}': {response.text.strip()}"
            )

    def ensure_index(
        self,
        index_name: str,
        dim: int,
        space_type: str = "cosine",
        precision: str = "float32",
    ) -> None:
        existing = None
        for item in self.list_indexes():
            if item.get("name") == index_name:
                existing = item
                break

        if existing is not None:
            existing_dim = int(existing.get("dimension", 0))
            if existing_dim and existing_dim != dim:
                raise EndeeClientError(
                    f"Index '{index_name}' exists with dimension {existing_dim}, "
                    f"but embedding model produces {dim}."
                )
            return

        self.create_index(
            index_name=index_name,
            dim=dim,
            space_type=space_type,
            precision=precision,
        )

    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> None:
        if not vectors:
            return

        self._request(
            "POST",
            f"/api/v1/index/{index_name}/vector/insert",
            json_body=vectors,
        )

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int,
    ) -> List[Dict[str, Any]]:
        response = self._request(
            "POST",
            f"/api/v1/index/{index_name}/search",
            json_body={"vector": query_vector, "k": k},
        )

        try:
            payload = msgpack.unpackb(
                response.content,
                raw=False,
                strict_map_key=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EndeeClientError(f"Could not decode Endee msgpack response: {exc}") from exc

        return self._parse_search_payload(payload)

    @staticmethod
    def _decode_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            try:
                return bytes(value).decode("utf-8", errors="replace")
            except ValueError:
                return ""
        return str(value)

    @staticmethod
    def _parse_search_payload(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            raw_results = payload.get("results", [])
        elif isinstance(payload, list):
            # ResultSet is msgpack-serialized as [results]
            if len(payload) == 1 and isinstance(payload[0], list):
                raw_results = payload[0]
            # HybridResultSet may come back as [dense_results, sparse_results]
            elif len(payload) == 2 and isinstance(payload[0], list):
                raw_results = payload[0]
            else:
                raw_results = payload
        else:
            raw_results = []

        parsed_results: List[Dict[str, Any]] = []
        for item in raw_results:
            parsed = EndeeClient._parse_vector_result(item)
            if parsed["id"]:
                parsed_results.append(parsed)

        return parsed_results

    @staticmethod
    def _parse_vector_result(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            similarity = float(item.get("similarity", 0.0))
            item_id = str(item.get("id", ""))
            meta = EndeeClient._decode_text(item.get("meta"))
            filter_text = EndeeClient._decode_text(item.get("filter"))
            vector = item.get("vector", [])
            return {
                "similarity": similarity,
                "id": item_id,
                "meta": meta,
                "filter": filter_text,
                "vector": vector,
            }

        if isinstance(item, (list, tuple)):
            values = list(item) + [None] * 6
            similarity = float(values[0] or 0.0)
            item_id = str(values[1] or "")
            meta = EndeeClient._decode_text(values[2])
            filter_text = EndeeClient._decode_text(values[3])
            vector = values[5] if isinstance(values[5], list) else []
            return {
                "similarity": similarity,
                "id": item_id,
                "meta": meta,
                "filter": filter_text,
                "vector": vector,
            }

        return {
            "similarity": 0.0,
            "id": "",
            "meta": "",
            "filter": "",
            "vector": [],
        }
